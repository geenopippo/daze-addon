"""Tests for signing in with an email address and a password.

The Daze portal signs users in through the Cognito sign-in page, and
``api/login.py`` walks that page the way a browser does. These tests
put a small stand-in for the page in front of the real module: it
answers the authorization request with a redirect and cookies, serves
the form, checks what is posted back (the hidden field, the cookies,
the PKCE verifier against the challenge it was given) and hands out a
code and tokens only when all of it adds up. A sign-in that works here
has therefore done every step, not just the last one.

The HTML is cut down from the real page: same form, same field names,
same wording when the password is wrong.

Home Assistant is not required.

Run with pytest, or standalone:

    python3 tests/test_login.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import sys
import types
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from aiohttp.client_exceptions import ClientConnectionError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_components" / "daze"


def _load_integration_modules() -> tuple[Any, Any, Any]:
    """Load the const, login, auth and api modules without Home Assistant.

    Returns:
        A tuple of the loaded ``login``, ``auth`` and ``api`` modules.

    """
    name = "daze_login_under_test"
    parent = types.ModuleType(name)
    parent.__path__ = [str(PACKAGE_DIR)]
    sys.modules[name] = parent

    const_spec = importlib.util.spec_from_file_location(
        f"{name}.const", PACKAGE_DIR / "const.py"
    )
    assert const_spec and const_spec.loader
    const_module = importlib.util.module_from_spec(const_spec)
    sys.modules[f"{name}.const"] = const_module
    const_spec.loader.exec_module(const_module)

    api_spec = importlib.util.spec_from_file_location(
        f"{name}.api",
        PACKAGE_DIR / "api" / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR / "api")],
    )
    assert api_spec and api_spec.loader
    api_module = importlib.util.module_from_spec(api_spec)
    sys.modules[f"{name}.api"] = api_module
    api_spec.loader.exec_module(api_module)

    return (
        sys.modules[f"{name}.api.login"],
        sys.modules[f"{name}.api.auth"],
        api_module,
    )


login, auth, api = _load_integration_modules()

HOST = "https://daze.auth.eu-central-1.amazoncognito.com"
PORTAL = "https://webportal.dazeservice.com/signin-oidc"
EMAIL = "driver@example.invalid"
PASSWORD = "correct horse battery staple"
SECRET_CODE = "one-time-code-9f1c"
TOKENS = {
    "id_token": "eyJ.FAKE-ID-TOKEN.0",
    "access_token": "eyJ.FAKE-ACCESS-TOKEN.1",
    "refresh_token": "eyJ.FAKE-REFRESH-TOKEN.2",
    "expires_in": 3600,
    "token_type": "Bearer",
}
# A value with the characters that would be quoted if the cookie were
# handed to aiohttp as a string.
XSRF = "Zm9v/YmFy+QUJD=="
CSRF_FIELD = "csrf-field-7d2e"


def _page(action: str, fields: str, message: str = "") -> str:
    """The sign-in page, cut down to what a client has to read."""
    return f"""<!DOCTYPE html><html><head><title>Sign-in</title>
<style>.x{{color:red}}</style></head><body>
<form method="post" action="{action.replace('&', '&amp;')}" id="primary-form">
<input type="hidden" name="csrf" value="{CSRF_FIELD}"/>
<h4>With existing account</h4>{message}
{fields}
<button type="submit">Sign in</button>
<input type="hidden" name="cognitoAsfData" value=""/>
</form><script>window.__x = "Incorrect username or password"</script>
</body></html>"""


BOTH_FIELDS = (
    '<input name="username" type="email" value=""/>'
    '<input name="password" type="password" value=""/>'
)
WRONG = (
    "<div>Invalid input: Incorrect username or password. , "
    "Invalid input: Incorrect username or password.</div>"
)


class FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(
        self,
        status: int,
        body: str = "",
        location: str = "",
        cookies: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = {"Location": location} if location else {}
        self.cookies: SimpleCookie = SimpleCookie()
        for name, value in (cookies or {}).items():
            # As a server sends them: the value exactly as given.
            self.cookies.load(f"{name}={value}")

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSignInPage:
    """The Cognito sign-in page, as far as a client can tell.

    ``variant`` picks what the page does once the right password is
    posted; a wrong one always gets the form back with the real page's
    wording.
    """

    def __init__(self, variant: str = "plain") -> None:
        self.variant = variant
        self.calls: list[dict[str, Any]] = []
        self.query: dict[str, str] = {}
        self.password_posts = 0

    # -- what aiohttp's session offers ------------------------------

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        assert kwargs.get("allow_redirects") is False, (
            "a redirect followed by aiohttp would lose the code"
        )
        parts = urlsplit(url)
        handler = {
            ("GET", "/oauth2/authorize"): self._authorize,
            ("GET", "/login"): self._form,
            ("POST", "/login"): self._submit,
            ("GET", "/mfa"): self._second_factor,
            ("POST", "/oauth2/token"): self._token,
        }.get((method, parts.path))
        assert handler is not None, f"unexpected request {method} {parts.path}"
        return handler(parts, kwargs)

    # -- the pages ---------------------------------------------------

    def _login_url(self) -> str:
        return "/login?" + urlencode(self.query)

    def _authorize(self, parts: Any, kwargs: dict[str, Any]) -> FakeResponse:
        self.query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        assert self.query["response_type"] == "code"
        assert self.query["code_challenge_method"] == "S256"
        assert self.query["redirect_uri"] == PORTAL
        assert self.query["scope"] == "openid profile email daze.api/daze.api"
        return FakeResponse(
            302,
            location=HOST + self._login_url(),
            cookies={"XSRF-TOKEN": XSRF, "lang": "en"},
        )

    def _form(self, parts: Any, kwargs: dict[str, Any]) -> FakeResponse:
        if self.variant == "down":
            return FakeResponse(503, "<html>Service Unavailable</html>")
        fields = BOTH_FIELDS
        if self.variant == "email first":
            fields = '<input name="username" type="email" value=""/>'
        return FakeResponse(200, _page(self._login_url(), fields))

    def _cookie_header(self, kwargs: dict[str, Any]) -> str:
        return kwargs["headers"].get("Cookie", "")

    def _submit(self, parts: Any, kwargs: dict[str, Any]) -> FakeResponse:
        data = kwargs["data"]
        assert data["csrf"] == CSRF_FIELD, "the hidden field was not sent back"
        assert f"XSRF-TOKEN={XSRF}" in self._cookie_header(kwargs), (
            "the cookie did not come back exactly as it was set: "
            + self._cookie_header(kwargs)
        )
        assert kwargs["headers"]["Origin"] == HOST

        if "password" not in data:
            # The page that asks for the email address first.
            assert data["username"] == EMAIL
            return FakeResponse(
                200,
                _page(
                    self._login_url(),
                    '<input name="password" type="password" value=""/>',
                ),
            )

        self.password_posts += 1
        if self.variant == "locked":
            return FakeResponse(
                400,
                _page(
                    self._login_url(),
                    BOTH_FIELDS,
                    "<div>Invalid input: Password attempts exceeded.</div>",
                ),
            )
        if data["password"] != PASSWORD or data.get("username", EMAIL) != EMAIL:
            return FakeResponse(400, _page(self._login_url(), BOTH_FIELDS, WRONG))

        if self.variant == "second factor":
            return FakeResponse(302, location="/mfa?" + urlencode(self.query))
        if self.variant == "denied":
            return FakeResponse(
                302,
                location=PORTAL + "?error=access_denied"
                "&error_description=User+is+not+allowed",
            )
        state = self.query["state"]
        if self.variant == "wrong state":
            state = "someone-elses"
        return FakeResponse(
            302,
            location=PORTAL + "?" + urlencode({"code": SECRET_CODE, "state": state}),
        )

    def _second_factor(self, parts: Any, kwargs: dict[str, Any]) -> FakeResponse:
        return FakeResponse(
            200,
            '<html><form method="post" action="/mfa">'
            '<input type="hidden" name="csrf" value="x"/>'
            '<input name="authentication_code" value=""/></form></html>',
        )

    def _token(self, parts: Any, kwargs: dict[str, Any]) -> FakeResponse:
        data = kwargs["data"]
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == SECRET_CODE
        assert data["redirect_uri"] == PORTAL
        challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(data["code_verifier"].encode()).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        assert challenge == self.query["code_challenge"], (
            "the verifier does not match the challenge sent at the start"
        )
        if self.variant == "no refresh token":
            return FakeResponse(
                200, json.dumps({"access_token": "a", "expires_in": 3600})
            )
        if self.variant == "token refused":
            return FakeResponse(400, json.dumps({"error": "invalid_grant"}))
        return FakeResponse(200, json.dumps(TOKENS))


def _sign_in(
    variant: str = "plain", password: str = PASSWORD
) -> tuple[Any, FakeSignInPage]:
    page = FakeSignInPage(variant)
    result = asyncio.run(login.async_login(page, EMAIL, password))
    return result, page


def _refused(variant: str = "plain", password: str = PASSWORD) -> Exception:
    page = FakeSignInPage(variant)
    try:
        asyncio.run(login.async_login(page, EMAIL, password))
    except Exception as err:  # noqa: BLE001 - the test inspects it
        return err
    raise AssertionError("expected the sign-in to fail")


# ------------------------------------------------------------------
# Signing in
# ------------------------------------------------------------------


def test_signing_in_returns_the_tokens_of_a_new_session() -> None:
    """Every step done right: the page hands out a code, then tokens."""
    result, page = _sign_in()

    assert result["access_token"] == TOKENS["access_token"]
    assert result["refresh_token"] == TOKENS["refresh_token"]
    assert result["expires_in"] == 3600
    assert [c["method"] for c in page.calls] == ["GET", "GET", "POST", "POST"]
    assert page.password_posts == 1


def test_the_email_address_and_password_go_in_the_form_only() -> None:
    """Posted as form fields, never in a URL where a proxy would log them."""
    _result, page = _sign_in()

    submit = page.calls[2]
    assert submit["data"]["username"] == EMAIL
    assert submit["data"]["password"] == PASSWORD
    assert all(PASSWORD not in call["url"] for call in page.calls)
    assert all(EMAIL not in call["url"] for call in page.calls)


def test_a_page_that_asks_for_the_email_address_first_still_works() -> None:
    """The same sign-in, split over two pages."""
    result, page = _sign_in("email first")

    assert result["refresh_token"] == TOKENS["refresh_token"]
    assert [c["method"] for c in page.calls] == [
        "GET",
        "GET",
        "POST",
        "POST",
        "POST",
    ]


def test_nothing_secret_is_logged_while_signing_in() -> None:
    """The log is what a user is asked to paste into an issue."""
    records: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(login.__name__)
    handler = Collect()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        _result, page = _sign_in()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    blob = "\n".join(records)
    assert records, "the test listened to the wrong logger"
    for secret in (
        PASSWORD,
        SECRET_CODE,
        TOKENS["access_token"],
        TOKENS["refresh_token"],
        XSRF,
        CSRF_FIELD,
        page.query["code_challenge"],
    ):
        assert secret not in blob, f"{secret[:8]}… reached the log"


# ------------------------------------------------------------------
# Not signing in
# ------------------------------------------------------------------


def test_a_wrong_password_is_reported_as_one() -> None:
    """With the page's own sentence, and after a single attempt."""
    page = FakeSignInPage()
    try:
        asyncio.run(login.async_login(page, EMAIL, "not the password"))
    except login.InvalidCredentialsError as err:
        assert "Incorrect username or password" in str(err)
    else:
        raise AssertionError("expected InvalidCredentialsError")

    assert page.password_posts == 1, "a refused password was sent again"


def test_text_inside_a_script_is_not_read_as_the_page_message() -> None:
    """The stand-in page carries the refusal wording in a script too.

    Only what a person would read counts: otherwise a correct password
    would be reported as wrong whenever the page ships its messages as
    data.
    """
    result, _page_ = _sign_in()

    assert result["access_token"] == TOKENS["access_token"]


def test_too_many_attempts_is_not_a_wrong_password() -> None:
    """Cognito asking to wait says nothing about the password.

    Reported as a wrong one, it would make the integration drop a
    password that is perfectly good.
    """
    err = _refused("locked")

    assert isinstance(err, login.LoginError)
    assert not isinstance(err, login.InvalidCredentialsError)
    assert "attempts exceeded" in str(err).lower()


def test_a_second_factor_is_something_only_a_browser_can_do() -> None:
    """The password was right and the page wants a code as well."""
    err = _refused("second factor")

    assert isinstance(err, login.LoginNotSupportedError)


def test_a_sign_in_page_that_is_down_is_a_temporary_failure() -> None:
    """No form to fill in: nothing was learnt about the credentials."""
    err = _refused("down")

    assert isinstance(err, login.LoginError)
    assert not isinstance(
        err, login.InvalidCredentialsError | login.LoginNotSupportedError
    )


def test_a_network_error_is_a_temporary_failure() -> None:
    """And is not allowed to escape as an aiohttp exception."""

    class Unreachable:
        def request(self, method: str, url: str, **kwargs: Any) -> Any:
            raise ClientConnectionError("no route")

    try:
        asyncio.run(login.async_login(Unreachable(), EMAIL, PASSWORD))
    except login.LoginError as err:
        assert "no route" in str(err)
        assert not isinstance(err, login.InvalidCredentialsError)
    else:
        raise AssertionError("expected LoginError")


def test_a_redirect_carrying_someone_elses_state_is_refused() -> None:
    """The check that ties the answer to the request that was made."""
    err = _refused("wrong state")

    assert isinstance(err, login.LoginError)
    assert "state" in str(err)


def test_an_error_from_the_portal_redirect_is_reported() -> None:
    """Cognito's own description is what explains it."""
    err = _refused("denied")

    assert isinstance(err, login.LoginError)
    assert "User is not allowed" in str(err)


def test_a_refused_code_exchange_is_a_failure_not_a_crash() -> None:
    err = _refused("token refused")

    assert isinstance(err, login.LoginError)
    assert "invalid_grant" in str(err)


def test_tokens_without_a_refresh_token_are_not_accepted() -> None:
    """An access token alone lasts an hour: not a session worth storing."""
    err = _refused("no refresh token")

    assert isinstance(err, login.LoginError)
    assert "refresh token" in str(err)


def _main() -> int:
    """Run every test in this module and report results."""
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except Exception as err:  # noqa: BLE001 - standalone runner
            failures += 1
            print(f"FAIL {test.__name__}: {type(err).__name__}: {err}")
        else:
            print(f"ok   {test.__name__}")

    print(f"\n{len(tests) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
