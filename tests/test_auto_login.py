"""Tests for signing in again when the refresh token has expired.

With pasted tokens, a refresh token Cognito refuses is the end of the
road until the user pastes new ones. With the email address and the
password stored as well, the auth client signs in again by itself —
carefully, because every attempt is a sign-in against the user's
account: a refused password is never sent twice, and an attempt that
could not be completed is repeated with a growing wait.

``api/login.py`` has its own tests; here it is replaced by a stand-in
so these can say exactly what the sign-in page answered.

Home Assistant is not required.

Run with pytest, or standalone:

    python3 tests/test_auto_login.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_components" / "daze"


def _load_integration_modules() -> tuple[Any, Any, Any]:
    """Load the const, login, auth and api modules without Home Assistant.

    Returns:
        A tuple of the loaded ``auth``, ``api`` and ``const`` modules.

    """
    name = "daze_auto_login_under_test"
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

    return sys.modules[f"{name}.api.auth"], api_module, const_module


auth, api, const = _load_integration_modules()

SOCKET_URL = "https://webapi.dazeservice.com/v3/sockets/SER1/remoteInfo"
EMAIL = "driver@example.invalid"
PASSWORD = "correct horse battery staple"
SIGNED_IN = {
    "access_token": "tok-signed-in",
    "refresh_token": "refresh-signed-in",
    "expires_in": 3600,
}
SOCKET_DATA = {"data": {"status": "idle"}}


class FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def json(self, content_type: str | None = "application/json") -> Any:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Records requests and replays queued responses, whatever the verb."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


class FakeSignIn:
    """Stands in for the sign-in page: answers what a test tells it to."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    async def __call__(
        self, session: Any, email: str, password: str
    ) -> dict[str, Any]:
        self.calls.append((email, password))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


UNAUTHORIZED = FakeResponse(401, {"message": "Unauthorized"})
REFUSED = FakeResponse(400, {"error": "invalid_grant"})


def _client(
    responses: list[Any], sign_in: FakeSignIn | None, **credentials: Any
) -> tuple[Any, Any, FakeSession, list[dict[str, Any]]]:
    """Build an API client whose auth client can sign in again."""
    stored: list[dict[str, Any]] = []
    session = FakeSession(responses)
    auth_client = auth.DazeAuthClient(
        "tok-old", "refresh-old", on_tokens=stored.append, **credentials
    )
    auth.async_login = sign_in
    return api.DazeApiClient(auth_client, session), auth_client, session, stored


def _raised(coro: Any) -> Exception:
    try:
        asyncio.run(coro)
    except Exception as err:  # noqa: BLE001 - the test inspects it
        return err
    raise AssertionError("expected the call to raise")


WITH_PASSWORD = {"username": EMAIL, "password": PASSWORD}


# ------------------------------------------------------------------
# The refresh token has expired
# ------------------------------------------------------------------


def test_an_expired_refresh_token_is_replaced_by_signing_in_again() -> None:
    """The whole point: the poll that hit the expiry still succeeds."""
    sign_in = FakeSignIn(SIGNED_IN)
    client, auth_client, session, stored = _client(
        [UNAUTHORIZED, REFUSED, FakeResponse(200, SOCKET_DATA)],
        sign_in,
        **WITH_PASSWORD,
    )

    result = asyncio.run(client._request("GET", SOCKET_URL))

    assert result == SOCKET_DATA
    assert sign_in.calls == [(EMAIL, PASSWORD)]
    assert session.calls[2]["headers"]["authorization"] == (
        "Bearer tok-signed-in"
    ), "the retry did not use the token of the new session"
    assert auth_client.access_token == "tok-signed-in"


def test_the_tokens_of_the_new_session_are_handed_over_to_be_stored() -> None:
    """Otherwise the next restart starts from the dead refresh token."""
    client, _auth_client, _session, stored = _client(
        [UNAUTHORIZED, REFUSED, FakeResponse(200, SOCKET_DATA)],
        FakeSignIn(SIGNED_IN),
        **WITH_PASSWORD,
    )

    asyncio.run(client._request("GET", SOCKET_URL))

    assert len(stored) == 1
    assert stored[0]["access_token"] == "tok-signed-in"
    assert stored[0]["refresh_token"] == "refresh-signed-in"


def test_without_a_password_the_user_is_asked_as_before() -> None:
    """Entries set up with pasted tokens behave exactly as they did."""
    sign_in = FakeSignIn(SIGNED_IN)
    client, _auth_client, _session, stored = _client(
        [UNAUTHORIZED, REFUSED], sign_in
    )

    err = _raised(client._request("GET", SOCKET_URL))

    assert isinstance(err, api.ApiAuthError)
    assert sign_in.calls == []
    assert stored == []


def test_a_refused_password_asks_the_user_and_is_never_sent_again() -> None:
    """A changed password must not be retried on every poll.

    Each try is a failed sign-in on the user's account, and Cognito
    locks an account after a handful of them.
    """
    sign_in = FakeSignIn(auth.InvalidCredentialsError("refused"))
    client, _auth_client, session, _stored = _client(
        [UNAUTHORIZED, REFUSED, UNAUTHORIZED, REFUSED],
        sign_in,
        **WITH_PASSWORD,
    )

    first = _raised(client._request("GET", SOCKET_URL))
    second = _raised(client._request("GET", SOCKET_URL))

    assert isinstance(first, api.ApiAuthError)
    assert isinstance(second, api.ApiAuthError)
    assert len(sign_in.calls) == 1, "a refused password was sent again"


def test_a_sign_in_only_a_browser_can_finish_asks_the_user() -> None:
    """A second factor will be asked for every time: do not insist."""
    sign_in = FakeSignIn(auth.LoginNotSupportedError("code"))
    client, _auth_client, _session, _stored = _client(
        [UNAUTHORIZED, REFUSED, UNAUTHORIZED, REFUSED],
        sign_in,
        **WITH_PASSWORD,
    )

    assert isinstance(
        _raised(client._request("GET", SOCKET_URL)), api.ApiAuthError
    )
    assert isinstance(
        _raised(client._request("GET", SOCKET_URL)), api.ApiAuthError
    )
    assert len(sign_in.calls) == 1


# ------------------------------------------------------------------
# The sign-in could not be completed
# ------------------------------------------------------------------


def test_an_unfinished_sign_in_is_an_ordinary_failed_poll() -> None:
    """The page was down: nothing says the password is wrong."""
    sign_in = FakeSignIn(auth.LoginError("page down"))
    client, _auth_client, _session, stored = _client(
        [UNAUTHORIZED, REFUSED], sign_in, **WITH_PASSWORD
    )

    err = _raised(client._request("GET", SOCKET_URL))

    assert isinstance(err, api.ApiError)
    assert not isinstance(err, api.ApiAuthError)
    assert stored == []


def test_the_next_polls_wait_instead_of_signing_in_again() -> None:
    """One attempt, then silence until the wait is over.

    Not even the refresh is asked for again: the refresh token is known
    to be refused, and asking every poll would only tell Cognito so
    another hundred times an hour.
    """
    sign_in = FakeSignIn(auth.LoginError("page down"))
    client, _auth_client, session, _stored = _client(
        [UNAUTHORIZED, REFUSED, UNAUTHORIZED, UNAUTHORIZED],
        sign_in,
        **WITH_PASSWORD,
    )

    _raised(client._request("GET", SOCKET_URL))
    calls_after_first = len(session.calls)

    for _ in range(2):
        err = _raised(client._request("GET", SOCKET_URL))
        assert isinstance(err, api.ApiError)
        assert not isinstance(err, api.ApiAuthError)

    assert len(sign_in.calls) == 1, "signed in again before the wait was over"
    assert len(session.calls) == calls_after_first + 2, (
        "Cognito was asked again during the wait"
    )


def test_after_the_wait_it_signs_in_and_carries_on() -> None:
    """The page comes back, and so does the integration, by itself."""
    sign_in = FakeSignIn(auth.LoginError("page down"), SIGNED_IN)
    client, auth_client, _session, stored = _client(
        [
            UNAUTHORIZED,
            REFUSED,
            UNAUTHORIZED,
            REFUSED,
            FakeResponse(200, SOCKET_DATA),
        ],
        sign_in,
        **WITH_PASSWORD,
    )

    _raised(client._request("GET", SOCKET_URL))
    assert auth_client._next_login_at > 0
    auth_client._next_login_at = 0.0  # the wait is over

    result = asyncio.run(client._request("GET", SOCKET_URL))

    assert result == SOCKET_DATA
    assert len(sign_in.calls) == 2
    assert stored[0]["refresh_token"] == "refresh-signed-in"
    assert auth_client._login_failures == 0


def test_the_wait_grows_with_each_failed_attempt() -> None:
    """Five minutes, ten, twenty: not a fixed beat against the page."""
    sign_in = FakeSignIn(auth.LoginError("page down"))
    client, auth_client, _session, _stored = _client(
        [UNAUTHORIZED, REFUSED] * 3, sign_in, **WITH_PASSWORD
    )

    waits = []
    for _ in range(3):
        before = auth.time.time()
        _raised(client._request("GET", SOCKET_URL))
        waits.append(round(auth_client._next_login_at - before))
        auth_client._next_login_at = 0.0

    base = const.LOGIN_RETRY_SECONDS
    assert waits == [base, base * 2, base * 4], waits


def test_after_enough_failed_attempts_the_user_is_asked() -> None:
    """A page that never answers properly has probably changed.

    Trying for ever would leave the wallbox unavailable with nothing
    telling the user why.
    """
    sign_in = FakeSignIn(auth.LoginError("page changed"))
    attempts = const.LOGIN_MAX_ATTEMPTS
    client, auth_client, _session, _stored = _client(
        [UNAUTHORIZED, REFUSED] * (attempts + 1), sign_in, **WITH_PASSWORD
    )

    errors = []
    for _ in range(attempts):
        errors.append(_raised(client._request("GET", SOCKET_URL)))
        auth_client._next_login_at = 0.0

    assert all(
        not isinstance(err, api.ApiAuthError) for err in errors[:-1]
    ), "gave up before the last attempt"
    assert isinstance(errors[-1], api.ApiAuthError)
    assert len(sign_in.calls) == attempts

    # And it stays given up: no further attempts by itself.
    assert isinstance(
        _raised(client._request("GET", SOCKET_URL)), api.ApiAuthError
    )
    assert len(sign_in.calls) == attempts


# ------------------------------------------------------------------
# An ordinary refresh
# ------------------------------------------------------------------


def test_a_rotated_refresh_token_is_handed_over_to_be_stored() -> None:
    """Cognito may answer a refresh with a new refresh token.

    The old one stops working, so the one on disk has to follow.
    """
    client, auth_client, _session, stored = _client(
        [
            UNAUTHORIZED,
            FakeResponse(
                200,
                {
                    "access_token": "tok-new",
                    "refresh_token": "refresh-rotated",
                    "expires_in": 3600,
                },
            ),
            FakeResponse(200, SOCKET_DATA),
        ],
        None,
    )

    asyncio.run(client._request("GET", SOCKET_URL))

    assert stored == [auth_client.async_get_tokens_for_store()]
    assert stored[0]["refresh_token"] == "refresh-rotated"


def test_an_hourly_refresh_writes_nothing() -> None:
    """Same refresh token as before: nothing worth a write to disk."""
    client, _auth_client, _session, stored = _client(
        [
            UNAUTHORIZED,
            FakeResponse(200, {"access_token": "tok-new", "expires_in": 3600}),
            FakeResponse(200, SOCKET_DATA),
        ],
        None,
    )

    asyncio.run(client._request("GET", SOCKET_URL))

    assert stored == []


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
