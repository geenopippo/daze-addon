"""Sign in to Daze with an email address and a password.

The Daze web portal does not ask Cognito for tokens directly. It sends
the browser to the Cognito sign-in page, which redirects back with a
one-time code that is then exchanged for tokens (the OAuth
authorization-code flow, with PKCE). This module walks the same pages a
browser would:

1. ``GET /oauth2/authorize`` redirects to the sign-in page and sets the
   cookies that page needs.
2. The sign-in page is an ordinary HTML form: a hidden ``csrf`` field,
   ``username`` and ``password``.
3. Posting it answers with a redirect to the portal carrying ``code``,
   or with the form again and a message when the credentials are wrong.
4. ``POST /oauth2/token`` exchanges the code for an access token and a
   refresh token: the same pair a user would otherwise copy out of the
   browser's local storage.

Nothing here depends on Home Assistant.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import secrets
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

from aiohttp import ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientError

from ..const import (
    CLIENT_ID,
    COGNITO_BASE_URL,
    LOGIN_REDIRECT_URI,
    OAUTH_SCOPE,
)

_LOGGER = logging.getLogger(__name__)

# The sign-in page is served to browsers; a client that does not look
# like one is not guaranteed the same HTML.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_TIMEOUT = ClientTimeout(total=30)
_REDIRECTS = (301, 302, 303, 307, 308)
_MAX_HOPS = 5
# One form for the credentials, with room for a sign-in page that asks
# for the email address first and the password on a second page.
_MAX_FORMS = 3

# What the sign-in page says when it sends the form back. English is
# requested explicitly, so these are stable.
_WRONG_CREDENTIALS = (
    "incorrect username or password",
    "user does not exist",
)
_TRY_LATER = (
    "attempts exceeded",
    "too many",
    "try again later",
    "limit exceeded",
)


class LoginError(Exception):
    """Raised when signing in did not produce tokens.

    On its own it means the attempt could not be completed — the
    network, an unexpected page, Cognito asking to wait — and says
    nothing about the credentials.
    """


class InvalidCredentialsError(LoginError):
    """Raised when Cognito refuses the email address and password."""


class LoginNotSupportedError(LoginError):
    """Raised when signing in needs a step this module cannot perform.

    A second factor, a forced password change, an unconfirmed account:
    things a person has to do in a browser.
    """


class _FormParser(HTMLParser):
    """Collect the forms of a page with their fields.

    Only what is needed to post one back: where it goes, how, and the
    name, type and value of each input.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Open a form, or record an input of the form being read."""
        values = {name: value or "" for name, value in attrs}

        if tag == "form":
            self._current = {
                "action": values.get("action", ""),
                "method": values.get("method", "get").lower(),
                "fields": {},
                "types": {},
            }
            self.forms.append(self._current)
            return

        if tag == "input" and self._current is not None:
            name = values.get("name")
            if name:
                self._current["fields"][name] = values.get("value", "")
                self._current["types"][name] = values.get(
                    "type", "text"
                ).lower()

    def handle_endtag(self, tag: str) -> None:
        """Close the form being read."""
        if tag == "form":
            self._current = None


def _sign_in_form(body: str) -> dict[str, Any] | None:
    """Return the form of a page that asks for credentials, if any.

    Args:
        body: The HTML of the page.

    Returns:
        The first POST form with a ``username`` or ``password`` field,
        or None when the page has none.

    """
    parser = _FormParser()
    try:
        parser.feed(body)
    except Exception:  # noqa: BLE001 - HTML that cannot be read is no form
        return None

    for form in parser.forms:
        if form["method"] != "post":
            continue
        if "password" in form["fields"] or "username" in form["fields"]:
            return form

    return None


def _visible_text(body: str) -> str:
    """Return the text a person would read on a page, on one line."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", body)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _page_message(body: str) -> str:
    """Return what the sign-in page says went wrong, if it says so.

    Args:
        body: The HTML of the page.

    Returns:
        A short sentence taken from the page, or an empty string.

    """
    text = _visible_text(body)
    match = re.search(r"Invalid input:\s*([^.,]{3,120})", text)
    if match:
        return match.group(1).strip()

    lowered = text.lower()
    for phrase in (*_WRONG_CREDENTIALS, *_TRY_LATER):
        position = lowered.find(phrase)
        if position >= 0:
            return text[position : position + len(phrase)]

    return ""


def _pkce_pair() -> tuple[str, str]:
    """Return a PKCE code verifier and its S256 challenge."""
    verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(48))
        .rstrip(b"=")
        .decode()
    )
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def _where(url: str) -> str:
    """Describe a URL for a log line without its query string.

    The query carries the one-time code and the PKCE challenge, neither
    of which belongs in a log.
    """
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path}"


async def _fetch(
    session: ClientSession,
    method: str,
    url: str,
    cookies: dict[str, str],
    *,
    data: dict[str, str] | None = None,
    referer: str | None = None,
) -> tuple[int, str, str]:
    """Make one request without following redirects.

    The cookies the server sets are added to ``cookies``, which is sent
    with every request: the sign-in page ties its form to them. They
    are sent back exactly as received, in a header built here, because
    handing them to aiohttp as strings would quote any value holding a
    ``/`` or an ``=``.

    Args:
        session: The aiohttp session to use.
        method: GET or POST.
        url: The absolute URL.
        cookies: The cookies collected so far; updated in place.
        data: Form fields for a POST.
        referer: The page the request comes from, for a POST.

    Returns:
        A tuple of the status, the ``Location`` header (empty when
        absent) and the body.

    Raises:
        LoginError: If the request fails at the network level.

    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en",
    }
    if referer:
        origin = urlsplit(referer)
        headers["Origin"] = f"{origin.scheme}://{origin.netloc}"
        headers["Referer"] = referer
    if cookies:
        headers["Cookie"] = "; ".join(
            f"{name}={value}" for name, value in cookies.items()
        )

    try:
        async with session.request(
            method,
            url,
            headers=headers,
            data=data,
            allow_redirects=False,
            timeout=_TIMEOUT,
        ) as response:
            for name, morsel in response.cookies.items():
                if morsel.coded_value in ("", '""'):
                    cookies.pop(name, None)  # the server cleared it
                else:
                    cookies[name] = morsel.coded_value
            location = response.headers.get("Location", "")
            body = await response.text()
            return response.status, location, body
    except (ClientError, TimeoutError) as err:
        detail = str(err) or type(err).__name__
        raise LoginError(
            f"Network error while signing in ({_where(url)}): {detail}"
        ) from err


def _code_from_redirect(location: str, state: str) -> str | None:
    """Return the one-time code if a redirect goes back to the portal.

    Args:
        location: The ``Location`` header of a redirect.
        state: The state sent with the authorization request.

    Returns:
        The code, or None if the redirect goes somewhere else.

    Raises:
        LoginError: If the portal is reached without a usable code.

    """
    if not location.startswith(LOGIN_REDIRECT_URI):
        return None

    query = parse_qs(urlsplit(location).query)

    if "error" in query:
        description = (query.get("error_description") or query["error"])[0]
        raise LoginError(f"Daze refused the sign-in: {description}")

    code = (query.get("code") or [""])[0]
    if not code:
        raise LoginError("The sign-in page redirected back without a code")

    if (query.get("state") or [""])[0] != state:
        raise LoginError(
            "The sign-in page redirected back with a different state"
        )

    return code


async def async_login(
    session: ClientSession, email: str, password: str
) -> dict[str, Any]:
    """Sign in with an email address and a password.

    Args:
        session: The aiohttp session to use.
        email: The email address of the Daze account.
        password: Its password.

    Returns:
        A dict with ``access_token``, ``refresh_token`` and
        ``expires_in``, and ``id_token`` when Cognito returns one.

    Raises:
        InvalidCredentialsError: If Cognito refuses the credentials.
        LoginNotSupportedError: If signing in needs a step that only a
            person in a browser can perform.
        LoginError: If the attempt could not be completed.

    """
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    cookies: dict[str, str] = {}

    url = f"{COGNITO_BASE_URL}/oauth2/authorize?" + urlencode(
        {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "scope": OAUTH_SCOPE,
            "redirect_uri": LOGIN_REDIRECT_URI,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    # 1. The authorization request, followed to the sign-in page.
    status, location, body = await _fetch(session, "GET", url, cookies)
    code: str | None = None
    hops = 0
    while status in _REDIRECTS and location and hops < _MAX_HOPS:
        code = _code_from_redirect(location, state)
        if code:
            break
        url = urljoin(url, location)
        status, location, body = await _fetch(session, "GET", url, cookies)
        hops += 1

    # 2. The form, posted back with the credentials.
    submitted = 0
    while code is None:
        form = _sign_in_form(body) if status < 500 else None

        if form is None:
            raise LoginError(
                f"Unexpected answer from the Daze sign-in page "
                f"(HTTP {status} at {_where(url)})"
            )

        if submitted >= _MAX_FORMS:
            raise LoginNotSupportedError(
                "The Daze sign-in page keeps asking for more than an "
                "email address and a password"
            )

        if submitted and "password" not in form["fields"]:
            # The email address was sent and the page did not move on
            # to the password: it wants something else.
            raise LoginNotSupportedError(
                "The Daze sign-in page asks for something other than "
                "an email address and a password"
            )

        fields = dict(form["fields"])
        if "username" in fields:
            fields["username"] = email
        if "password" in fields:
            fields["password"] = password
        sent_password = "password" in fields

        page = url
        url = urljoin(page, form["action"] or page)
        _LOGGER.debug("Posting the Daze sign-in form to %s", _where(url))
        status, location, body = await _fetch(
            session, "POST", url, cookies, data=fields, referer=page
        )
        submitted += 1

        hops = 0
        while status in _REDIRECTS and location and hops < _MAX_HOPS:
            code = _code_from_redirect(location, state)
            if code:
                break
            url = urljoin(url, location)
            status, location, body = await _fetch(session, "GET", url, cookies)
            hops += 1

        if code:
            break

        if not sent_password:
            # A page that asked for the email address only: whatever
            # came back is read at the top of the loop.
            continue

        # The password was sent and no code came back.
        message = _page_message(body)
        lowered = message.lower()

        if any(phrase in lowered for phrase in _TRY_LATER):
            raise LoginError(f"Daze asks to wait before signing in: {message}")

        again = _sign_in_form(body)
        asks_again = again is not None and "password" in again["fields"]

        if asks_again or any(
            phrase in lowered for phrase in _WRONG_CREDENTIALS
        ):
            # Being asked for the password again is a refusal whatever
            # the wording. Reading it as one also keeps a stored
            # password from being retried against an account that
            # locks after a few wrong attempts.
            raise InvalidCredentialsError(
                message or "Daze refused the email address and password"
            )

        if status in (200, 400):
            # The password was accepted and the page wants more: a
            # verification code, a new password, a confirmation.
            raise LoginNotSupportedError(
                "Signing in to Daze needs a further step in a browser"
                + (f": {message}" if message else "")
            )

        raise LoginError(
            f"Unexpected answer from the Daze sign-in page "
            f"(HTTP {status} at {_where(url)})"
            + (f": {message}" if message else "")
        )

    # 3. The code, exchanged for tokens.
    token_url = f"{COGNITO_BASE_URL}/oauth2/token"
    status, _location, body = await _fetch(
        session,
        "POST",
        token_url,
        cookies,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": LOGIN_REDIRECT_URI,
            "code_verifier": verifier,
        },
    )

    if status != 200:
        # The body of a refusal is Cognito's error name, nothing secret.
        raise LoginError(
            f"Daze did not exchange the sign-in code for tokens "
            f"(HTTP {status}): {body[:200]}"
        )

    try:
        payload = json.loads(body)
    except ValueError as err:
        raise LoginError("Daze answered the token request with no JSON") from err

    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise LoginError("Daze answered the token request without tokens")

    if not payload.get("refresh_token"):
        raise LoginError(
            "Daze answered the token request without a refresh token"
        )

    _LOGGER.debug("Signed in to Daze with email address and password")

    return {
        "access_token": payload["access_token"],
        "refresh_token": payload["refresh_token"],
        "id_token": payload.get("id_token"),
        "expires_in": payload.get("expires_in", 3600),
    }
