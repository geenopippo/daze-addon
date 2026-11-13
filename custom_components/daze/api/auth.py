"""Daze OAuth 2.0 authentication client for Amazon Cognito."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError

from ..const import (
    CLIENT_ID,
    COGNITO_BASE_URL,
    COGNITO_IDP_URL,
    DEFAULT_TOKEN_EXPIRY_BUFFER,
    GET_USER_CONTENT_TYPE,
    GET_USER_TARGET,
    LOGIN_MAX_ATTEMPTS,
    LOGIN_RETRY_MAX_SECONDS,
    LOGIN_RETRY_SECONDS,
    REDIRECT_URI,
)
from .login import (
    InvalidCredentialsError,
    LoginError,
    LoginNotSupportedError,
    async_login,
)

_LOGGER = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when authentication fails (invalid/expired tokens, network error)."""


class RefreshTokenRejectedError(AuthError):
    """Raised when Cognito itself refuses the refresh token.

    This is the one refresh failure that new tokens from the user can
    fix, so it is the only one that should start re-authentication. A
    refresh that fails for any other reason — the network, a timeout, a
    5xx from Cognito — says nothing about the token and is worth trying
    again on the next poll.

    Subclasses AuthError so callers that do not care about the
    difference keep working.
    """


async def async_fetch_user(
    session: ClientSession, access_token: str
) -> tuple[int, dict[str, Any]]:
    """Call the Cognito user pool GetUser operation.

    This replaces the hosted-UI ``/oauth2/userInfo`` endpoint, which
    requires the ``openid`` scope. The Daze web portal issues access
    tokens scoped ``aws.cognito.signin.user.admin`` without ``openid``,
    so userInfo rejects every token a user can obtain. GetUser accepts
    that scope and returns the same profile attributes.

    Two differences from userInfo matter to callers:

    - The access token is sent in the request body, not in an
      ``Authorization`` header.
    - An invalid or expired token yields HTTP 400 with a
      ``NotAuthorizedException`` type, not HTTP 401.

    Args:
        session: An aiohttp ClientSession to use for the request.
        access_token: The Cognito access token to authenticate with.

    Returns:
        A tuple of the HTTP status code and the parsed JSON body. The
        body is an empty dict if the response was not valid JSON.

    Raises:
        AuthError: If the request fails at the network level.

    """
    headers = {
        "Content-Type": GET_USER_CONTENT_TYPE,
        "X-Amz-Target": GET_USER_TARGET,
    }

    try:
        async with session.post(
            COGNITO_IDP_URL,
            headers=headers,
            json={"AccessToken": access_token},
        ) as response:
            # Cognito replies with application/x-amz-json-1.1, which
            # aiohttp refuses to decode unless content_type is relaxed.
            try:
                body = await response.json(content_type=None)
            except (ValueError, TypeError):
                body = {}

            if not isinstance(body, dict):
                body = {}

            if response.status == 200:
                return response.status, body

            # Some Daze portal accounts receive access tokens scoped
            # "openid profile email daze.api/daze.api" without
            # "aws.cognito.signin.user.admin". GetUser rejects those
            # ("Access Token does not have required scopes") while the
            # hosted-UI userInfo endpoint accepts them, so fall back
            # to userInfo and reshape its reply like GetUser's so
            # callers do not need to change.
            _LOGGER.debug(
                "GetUser rejected the token (%s), trying userInfo",
                describe_get_user_error(response.status, body),
            )
            try:
                async with session.get(
                    f"{COGNITO_BASE_URL}/oauth2/userInfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                ) as info_response:
                    try:
                        info = await info_response.json(content_type=None)
                    except (ValueError, TypeError):
                        info = {}
                    if info_response.status == 200 and isinstance(info, dict):
                        return 200, {
                            "Username": info.get("username", info.get("sub", "")),
                            "UserAttributes": [
                                {"Name": key, "Value": value}
                                for key, value in info.items()
                                if isinstance(value, str)
                            ],
                        }
            except ClientError as err:
                _LOGGER.warning("Network error during userInfo: %s", err)
                raise AuthError(f"Network error during userInfo: {err}") from err

            # userInfo returned non-200 — fall through to return the
            # original GetUser failure.

        # Either userInfo returned non-200 or its inner async with
        # completed without raising.  In both cases return the
        # original GetUser status/body so callers get the failure.

        return response.status, body

    except ClientError as err:
        _LOGGER.warning("Network error during GetUser: %s", err)
        raise AuthError(f"Network error during GetUser: {err}") from err


def describe_get_user_error(status: int, body: dict[str, Any]) -> str:
    """Summarise a failed GetUser response without leaking the body.

    Args:
        status: The HTTP status code returned by Cognito.
        body: The parsed JSON body.

    Returns:
        A short, safe description for logs and error messages.

    """
    error_type = body.get("__type") or "unknown error"
    return f"HTTP {status}: {error_type}"


class DazeAuthClient:
    """Manages Daze Cognito OAuth token lifecycle.

    Stores the access and refresh tokens, checks expiry, performs token
    refresh, and validates tokens against the Cognito userInfo endpoint.

    Given the email address and password of the account as well, it
    signs in again by itself when Cognito refuses the refresh token,
    which is otherwise the moment the user has to paste new tokens.
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        token_expiry: float | None = None,
        *,
        username: str | None = None,
        password: str | None = None,
        on_tokens: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialise the auth client.

        Args:
            access_token: The Cognito access token (Bearer token).
            refresh_token: The Cognito refresh token.
            token_expiry: Optional UNIX timestamp of the access token expiry.
            username: Optional email address of the Daze account.
            password: Optional password of the Daze account.
            on_tokens: Optional callback, called with the new tokens
                whenever the refresh token changes (a sign-in, or
                Cognito rotating it) so the caller can store them.

        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expiry = token_expiry
        self._username = username
        self._password = password
        self._on_tokens = on_tokens
        # Bookkeeping for signing in again after a refused refresh.
        self._refresh_refused = False
        self._credentials_refused = False
        self._login_failures = 0
        self._next_login_at = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_headers(self) -> dict[str, str]:
        """Return the authorisation header dict for API calls."""
        return {"authorization": f"Bearer {self._access_token}"}

    @property
    def has_credentials(self) -> bool:
        """Return True if an email address and a password are stored."""
        return bool(self._username and self._password)

    def _store(self, payload: dict[str, Any]) -> None:
        """Take the tokens of a refresh or a sign-in.

        The caller is told only when the refresh token changes: that is
        the one worth writing down, since the access token is replaced
        every hour and can always be had again from it.
        """
        self._access_token = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 3600)

        refresh_token = payload.get("refresh_token")
        if refresh_token and refresh_token != self._refresh_token:
            self._refresh_token = refresh_token
            if self._on_tokens is not None:
                self._on_tokens(self.async_get_tokens_for_store())

    async def async_login(self, session: ClientSession) -> str:
        """Sign in with the stored email address and password.

        Args:
            session: An aiohttp ClientSession to use for the requests.

        Returns:
            The new access token string.

        Raises:
            InvalidCredentialsError: If Cognito refuses the credentials.
            LoginNotSupportedError: If signing in needs a step that only
                a person in a browser can perform.
            LoginError: If no credentials are stored, or the attempt
                could not be completed.

        """
        if not self.has_credentials:
            raise LoginError("No email address and password are stored")

        payload = await async_login(
            session,
            self._username,  # type: ignore[arg-type]
            self._password,  # type: ignore[arg-type]
        )
        self._store(payload)
        self._refresh_refused = False
        self._credentials_refused = False
        self._login_failures = 0
        self._next_login_at = 0.0
        return self._access_token

    async def _async_login_after_refusal(
        self, session: ClientSession, refusal: RefreshTokenRejectedError
    ) -> str:
        """Sign in again because Cognito refused the refresh token.

        A refused password is never tried a second time, and an attempt
        that could not be completed is repeated a few times with a
        growing wait: the alternative is a failed sign-in against the
        user's account every poll.

        Args:
            session: An aiohttp ClientSession to use for the requests.
            refusal: The rejection that brought us here, raised again
                when signing in cannot fix it.

        Returns:
            The new access token string.

        Raises:
            RefreshTokenRejectedError: If signing in is not possible or
                was refused: the user has to act.
            AuthError: If the attempt could not be completed and will
                be made again later.

        """
        if self._credentials_refused:
            raise refusal

        try:
            access_token = await self.async_login(session)
        except (InvalidCredentialsError, LoginNotSupportedError) as err:
            self._credentials_refused = True
            _LOGGER.warning(
                "Daze did not accept the stored email address and "
                "password: %s",
                err,
            )
            raise refusal from err
        except LoginError as err:
            self._login_failures += 1
            if self._login_failures >= LOGIN_MAX_ATTEMPTS:
                self._credentials_refused = True
                _LOGGER.warning(
                    "Signing in to Daze failed %d times, giving up: %s",
                    self._login_failures,
                    err,
                )
                raise refusal from err

            wait = min(
                LOGIN_RETRY_SECONDS * 2 ** (self._login_failures - 1),
                LOGIN_RETRY_MAX_SECONDS,
            )
            self._next_login_at = time.time() + wait
            _LOGGER.warning(
                "Signing in to Daze did not complete, next attempt in "
                "%d minutes: %s",
                wait // 60,
                err,
            )
            raise AuthError(
                f"Signing in to Daze did not complete: {err}"
            ) from err

        _LOGGER.info(
            "The Daze refresh token had expired: signed in again with "
            "the stored email address and password"
        )
        return access_token

    def is_token_expired(self) -> bool:
        """Return True if the access token is expired or near expiry.

        Uses a 60-second buffer before the actual expiry time to allow
        for clock skew and request latency.
        """
        if self._token_expiry is None:
            return False
        return time.time() >= (self._token_expiry - DEFAULT_TOKEN_EXPIRY_BUFFER)

    async def async_refresh_access_token(
        self, session: ClientSession
    ) -> str:
        """Refresh the access token via Cognito OAuth2 token endpoint.

        Args:
            session: An aiohttp ClientSession to use for the request.

        Returns:
            The new access token string.

        Raises:
            RefreshTokenRejectedError: If Cognito answers that the
                refresh token is expired, revoked or otherwise invalid,
                and signing in again with a stored email address and
                password is not possible or was refused too.
            AuthError: If the refresh could not be completed for any
                other reason (network error, timeout, unexpected
                status). The refresh token may still be good.

        """
        if self._refresh_refused and time.time() < self._next_login_at:
            # The refresh token is known to be refused and the last
            # attempt to sign in again did not complete: wait before
            # asking Cognito anything else.
            raise AuthError(
                "Signing in to Daze again is postponed after a failed attempt"
            )

        url = f"{COGNITO_BASE_URL}/oauth2/token"
        data = {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }

        _LOGGER.debug("Refreshing access token via Cognito")

        refusal: RefreshTokenRejectedError

        try:
            async with session.post(
                url, headers=headers, data=data
            ) as response:
                if response.status == 200:
                    payload: dict[str, Any] = await response.json()
                    self._refresh_refused = False
                    self._store(payload)

                    _LOGGER.debug(
                        "Access token refreshed successfully "
                        "(expires in %ss)",
                        payload.get("expires_in", 3600),
                    )

                    return self._access_token

                body = await response.text()

                if response.status not in (400, 401):
                    # Not an answer about the token: Cognito is
                    # throttling or unwell. Logged as a warning because
                    # the caller retries, and an error here would read
                    # as something the user has to act on.
                    _LOGGER.warning(
                        "Unexpected token refresh response (HTTP %s): %s",
                        response.status,
                        body,
                    )
                    raise AuthError(
                        f"Token refresh unexpected status {response.status}: "
                        f"{body}"
                    )

                _LOGGER.warning(
                    "Token refresh failed (HTTP %s): %s",
                    response.status,
                    body,
                )
                refusal = RefreshTokenRejectedError(
                    f"Token refresh failed with status {response.status}: "
                    f"{body}"
                )

        except AuthError:
            raise
        except (ClientError, TimeoutError) as err:
            # TimeoutError is what aiohttp raises when the session
            # timeout expires; it is not a ClientError.
            detail = str(err) or type(err).__name__
            _LOGGER.warning("Network error during token refresh: %s", detail)
            raise AuthError(
                f"Network error during token refresh: {detail}"
            ) from err

        # Cognito refused the refresh token. With an email address and
        # a password stored this is no longer the user's problem: sign
        # in again, outside the request that has just been answered.
        self._refresh_refused = True

        if not self.has_credentials:
            raise refusal

        return await self._async_login_after_refusal(session, refusal)

    async def async_validate_tokens(
        self, session: ClientSession
    ) -> bool:
        """Validate the stored access token against Cognito GetUser.

        Args:
            session: An aiohttp ClientSession to use for the request.

        Returns:
            True if the token is valid (HTTP 200).

        Raises:
            AuthError: If the token is invalid or a network error occurs.

        """
        _LOGGER.debug("Validating tokens via Cognito GetUser")

        status, body = await async_fetch_user(session, self._access_token)

        if status == 200:
            return True

        detail = describe_get_user_error(status, body)
        _LOGGER.warning("Token validation failed (%s)", detail)
        raise AuthError(f"Token validation failed, {detail}")

    @property
    def access_token(self) -> str:
        """Return the current access token."""
        return self._access_token

    @property
    def token_expiry(self) -> float | None:
        """Return the access token expiry UNIX timestamp."""
        return self._token_expiry

    def async_get_tokens_for_store(self) -> dict[str, Any]:
        """Return a dict safe for config entry storage.

        This does NOT include raw API responses or sensitive metadata.
        """
        return {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "token_expiry": self._token_expiry,
        }
