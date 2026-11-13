"""Tests for what a failed token refresh, or a 401 after one, means.

Re-authentication stops the polling until the user pastes new tokens,
so it must be asked for only when new tokens would help: when Cognito
itself refuses the refresh token. Everything else — a refresh that did
not get through, a 5xx from Cognito, an API that answers 401 to a token
Cognito has just issued — is a temporary failure the next poll retries.

Like test_auth_getuser.py, these import the real auth and API modules.
Home Assistant is not required.

Run with pytest, or standalone:

    python3 tests/test_token_refresh.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

from aiohttp.client_exceptions import ClientConnectionError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_components" / "daze"


def _load_integration_modules() -> tuple[Any, Any]:
    """Load the const, auth and api modules without Home Assistant.

    Returns:
        A tuple of the loaded ``auth`` and ``api`` modules.

    """
    parent = types.ModuleType("daze_refresh_under_test")
    parent.__path__ = [str(PACKAGE_DIR)]
    sys.modules["daze_refresh_under_test"] = parent

    const_spec = importlib.util.spec_from_file_location(
        "daze_refresh_under_test.const", PACKAGE_DIR / "const.py"
    )
    assert const_spec and const_spec.loader
    const_module = importlib.util.module_from_spec(const_spec)
    sys.modules["daze_refresh_under_test.const"] = const_module
    const_spec.loader.exec_module(const_module)

    api_spec = importlib.util.spec_from_file_location(
        "daze_refresh_under_test.api",
        PACKAGE_DIR / "api" / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR / "api")],
    )
    assert api_spec and api_spec.loader
    api_module = importlib.util.module_from_spec(api_spec)
    sys.modules["daze_refresh_under_test.api"] = api_module
    api_spec.loader.exec_module(api_module)

    return sys.modules["daze_refresh_under_test.api.auth"], api_module


auth, api = _load_integration_modules()

# Keep the suite fast; the wall-clock waits are not under test.
api.COMMAND_RETRY_DELAY = 0.0
api.COMMAND_RETRY_MAX_DELAY = 0.0

SOCKET_URL = "https://webapi.dazeservice.com/v3/sockets/SER1/remoteInfo"
COMMAND_URL = "https://webapi.dazeservice.com/v3/sockets/SER1/commands/playcharge"


# ------------------------------------------------------------------
# Fake aiohttp session
# ------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def json(self, content_type: str | None = "application/json") -> Any:
        """Return the canned payload, ignoring content type."""
        return self._payload

    async def text(self) -> str:
        """Return the body as JSON text, as aiohttp does."""
        return json.dumps(self._payload)

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FailingRequest:
    """A request that never gets an answer: entering it raises."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self) -> FakeResponse:
        raise self._error

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Records requests and replays queued responses, whatever the verb."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        """Record a POST and return the next queued response."""
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self._responses.pop(0)

    def get(self, url: str, **kwargs: Any) -> Any:
        """Record a GET and return the next queued response."""
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self._responses.pop(0)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Record a generic request and return the next queued response."""
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._responses.pop(0)


UNAUTHORIZED = FakeResponse(401, {"message": "Unauthorized"})
REFRESHED = {"access_token": "tok-new", "expires_in": 3600}
REFUSED = {"error": "invalid_grant"}
SOCKET_DATA = {"data": {"status": "idle"}}


def _client(responses: list[Any]) -> tuple[Any, FakeSession]:
    """Build an API client over a session that replays ``responses``."""
    session = FakeSession(responses)
    client = api.DazeApiClient(
        auth.DazeAuthClient("tok-old", "refresh-123"), session
    )
    return client, session


def _raised(coro: Any) -> Exception:
    """Run a coroutine that must raise, and return what it raised."""
    try:
        asyncio.run(coro)
    except Exception as err:  # noqa: BLE001 - the test inspects it
        return err
    raise AssertionError("expected the call to raise")


# ------------------------------------------------------------------
# The refresh itself
# ------------------------------------------------------------------


def test_refresh_refused_by_cognito_is_a_rejection() -> None:
    """400 and 401 are Cognito's answer about the token: re-auth material."""
    for status in (400, 401):
        session = FakeSession([FakeResponse(status, REFUSED)])
        client = auth.DazeAuthClient("tok-old", "refresh-123")

        err = _raised(client.async_refresh_access_token(session))

        assert isinstance(err, auth.RefreshTokenRejectedError), status
        # Still an AuthError, for callers that do not tell them apart.
        assert isinstance(err, auth.AuthError)


def test_refresh_server_error_is_not_a_rejection() -> None:
    """A 5xx or a throttle says nothing about the refresh token."""
    for status in (429, 500, 503):
        session = FakeSession([FakeResponse(status, {"message": "busy"})])
        client = auth.DazeAuthClient("tok-old", "refresh-123")

        err = _raised(client.async_refresh_access_token(session))

        assert isinstance(err, auth.AuthError), status
        assert not isinstance(err, auth.RefreshTokenRejectedError), status
        # The old token is kept: nothing was issued.
        assert client.access_token == "tok-old"


def test_refresh_that_never_gets_through_is_not_a_rejection() -> None:
    """A dropped connection or a timeout must not look like a bad token."""
    for error in (ClientConnectionError("no route"), TimeoutError()):
        session = FakeSession([FailingRequest(error)])
        client = auth.DazeAuthClient("tok-old", "refresh-123")

        err = _raised(client.async_refresh_access_token(session))

        assert isinstance(err, auth.AuthError), type(error).__name__
        assert not isinstance(err, auth.RefreshTokenRejectedError)
        # A timeout has an empty message; the reason must still be named.
        assert type(error).__name__ in str(err) or "no route" in str(err)


# ------------------------------------------------------------------
# A 401 from the API
# ------------------------------------------------------------------


def test_401_then_refresh_then_success_uses_the_new_token() -> None:
    """The ordinary case: an expired access token, refreshed and retried."""
    client, session = _client(
        [UNAUTHORIZED, FakeResponse(200, REFRESHED), FakeResponse(200, SOCKET_DATA)]
    )

    result = asyncio.run(client._request("GET", SOCKET_URL))

    assert result == SOCKET_DATA
    assert session.calls[0]["headers"]["authorization"] == "Bearer tok-old"
    assert session.calls[1]["url"].endswith("/oauth2/token")
    assert session.calls[2]["headers"]["authorization"] == "Bearer tok-new"


def test_refresh_refused_asks_for_re_authentication() -> None:
    """The one case new tokens fix: Cognito refuses the refresh token."""
    client, _session = _client([UNAUTHORIZED, FakeResponse(400, REFUSED)])

    err = _raised(client._request("GET", SOCKET_URL))

    assert isinstance(err, api.ApiAuthError)


def test_refresh_that_did_not_complete_is_retried_not_re_authenticated() -> None:
    """Network trouble during the hourly refresh must not stop the polling.

    Before, any failed refresh raised ApiAuthError: a router restart at
    the wrong moment, or Home Assistant starting before the network was
    up, left the integration waiting for tokens that were still good.
    """
    for failure in (
        FakeResponse(503, {"message": "Service Unavailable"}),
        FailingRequest(ClientConnectionError("no route")),
        FailingRequest(TimeoutError()),
    ):
        client, session = _client([UNAUTHORIZED, failure])

        err = _raised(client._request("GET", SOCKET_URL))

        assert isinstance(err, api.ApiError), type(failure).__name__
        assert not isinstance(err, api.ApiAuthError), type(failure).__name__
        assert len(session.calls) == 2  # no retry of the request itself


def test_401_after_a_successful_refresh_is_a_temporary_failure() -> None:
    """A token Cognito has just issued, refused: not the user's to fix.

    This is the failure that was observed: the API answered 401, the
    refresh succeeded, the retry was answered 401 again, and the
    integration asked for re-authentication with tokens that worked
    unchanged after the next restart.
    """
    client, session = _client(
        [UNAUTHORIZED, FakeResponse(200, REFRESHED), UNAUTHORIZED]
    )

    err = _raised(client._request("GET", SOCKET_URL))

    assert isinstance(err, api.ApiError)
    assert not isinstance(err, api.ApiAuthError)
    assert err.status == 401
    assert len(session.calls) == 3


def test_the_poll_after_a_double_401_carries_on_normally() -> None:
    """Nothing is left behind: the next request works with the new token."""
    client, session = _client(
        [
            UNAUTHORIZED,
            FakeResponse(200, REFRESHED),
            UNAUTHORIZED,
            FakeResponse(200, SOCKET_DATA),
        ]
    )

    _raised(client._request("GET", SOCKET_URL))
    result = asyncio.run(client._request("GET", SOCKET_URL))

    assert result == SOCKET_DATA
    assert session.calls[3]["headers"]["authorization"] == "Bearer tok-new"


def test_a_command_is_not_repeated_on_a_double_401() -> None:
    """Commands retry the RPC failure only; a 401 is reported at once."""
    client, session = _client(
        [UNAUTHORIZED, FakeResponse(200, REFRESHED), UNAUTHORIZED]
    )

    err = _raised(client._post_command(COMMAND_URL, {}, attempts=8))

    assert isinstance(err, api.ApiError)
    assert not isinstance(err, api.ApiAuthError)
    assert len(session.calls) == 3


def test_a_command_still_asks_for_re_authentication_when_refused() -> None:
    """The command path keeps passing a real rejection through."""
    client, _session = _client([UNAUTHORIZED, FakeResponse(401, REFUSED)])

    err = _raised(client._post_command(COMMAND_URL, {}, attempts=8))

    assert isinstance(err, api.ApiAuthError)


# ------------------------------------------------------------------
# GetUser, used by the config flow
# ------------------------------------------------------------------

GET_USER_DENIED = {
    "__type": "NotAuthorizedException",
    "message": "Invalid Access Token",
}


def test_get_user_info_refresh_trouble_is_not_an_auth_failure() -> None:
    """During setup too, an unfinished refresh is a network problem."""
    client, _session = _client(
        [
            FakeResponse(400, GET_USER_DENIED),  # GetUser
            FakeResponse(400, {"error": "unauthorized"}),  # userInfo fallback
            FailingRequest(ClientConnectionError("no route")),  # refresh
        ]
    )

    err = _raised(client.async_get_user_info())

    assert isinstance(err, api.ApiError)
    assert not isinstance(err, api.ApiAuthError)


def test_get_user_info_refresh_refused_is_an_auth_failure() -> None:
    """And a refused refresh token still is one."""
    client, _session = _client(
        [
            FakeResponse(400, GET_USER_DENIED),
            FakeResponse(400, {"error": "unauthorized"}),
            FakeResponse(400, REFUSED),
        ]
    )

    err = _raised(client.async_get_user_info())

    assert isinstance(err, api.ApiAuthError)


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
