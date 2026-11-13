"""Config flow for Daze Wallbox integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import DazeApiClient
from .api.auth import AuthError, DazeAuthClient
from .api.login import (
    InvalidCredentialsError,
    LoginError,
    LoginNotSupportedError,
    async_login,
)
from .const import (
    CONF_ACCESS_TOKEN,
    CONF_DEVICE_PROFILE,
    CONF_EMAIL,
    CONF_EVSE_NAME,
    CONF_FIRMWARE_VERSION,
    CONF_GRID_POWER_SENSOR,
    CONF_NETWORK_NAME,
    CONF_NETWORK_UID,
    CONF_PASSWORD,
    CONF_POLL_INTERVAL,
    CONF_REFRESH_TOKEN,
    CONF_SERIAL_NUMBER,
    CONF_SOFTWARE_VERSION,
    CONF_SOLAR_RESERVE,
    CONF_SUPPLY_PHASES,
    CONF_USERNAME,
    DEFAULT_POLL_INTERVAL,
    DOMAIN,
    MAX_POLL_INTERVAL,
    MIN_POLL_INTERVAL,
    SUPPLY_PHASES_SINGLE,
    SUPPLY_PHASES_THREE,
)
from .payload import device_name

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ACCESS_TOKEN): str,
        vol.Required(CONF_REFRESH_TOKEN): str,
    }
)


async def _validate_tokens(
    hass: HomeAssistant, access_token: str, refresh_token: str
) -> dict[str, Any]:
    """Validate tokens and return user info.

    Returns a dict with keys:
        - "email": the user's email address
        - "access_token", "refresh_token" (passed through)

    Raises vol.Invalid if validation fails.
    """
    session = async_get_clientsession(hass)
    auth_client = DazeAuthClient(access_token, refresh_token)

    try:
        await auth_client.async_validate_tokens(session)
    except AuthError as err:
        _LOGGER.warning("Token validation failed: %s", err)
        raise vol.Invalid("invalid_token") from err

    # Fetch user info to get the email address
    api = DazeApiClient(auth_client, session)
    user_info = await api.async_get_user_info()

    return {
        "email": user_info.get("email", ""),
        CONF_ACCESS_TOKEN: access_token,
        CONF_REFRESH_TOKEN: refresh_token,
    }


async def _sign_in(
    hass: HomeAssistant, email: str, password: str
) -> dict[str, Any]:
    """Sign in with an email address and a password.

    Returns the tokens of the new session: ``access_token`` and
    ``refresh_token``, the same pair the token form asks the user to
    copy out of the browser.

    Raises vol.Invalid, carrying the error key to show, if signing in
    fails. The three keys tell apart the things the user can do about
    it: retype the password, use the tokens instead, or try later.
    """
    session = async_get_clientsession(hass)

    try:
        return await async_login(session, email, password)
    except InvalidCredentialsError as err:
        raise vol.Invalid("invalid_auth") from err
    except LoginNotSupportedError as err:
        _LOGGER.warning("Signing in to Daze needs a browser: %s", err)
        raise vol.Invalid("login_not_supported") from err
    except LoginError as err:
        _LOGGER.warning("Signing in to Daze did not complete: %s", err)
        raise vol.Invalid("login_failed") from err


async def _fetch_networks(
    hass: HomeAssistant,
    api_client: DazeApiClient,
    email: str,
) -> list[dict[str, Any]]:
    """Fetch available networks for the user."""
    try:
        return await api_client.async_get_networks(email)
    except Exception as err:
        _LOGGER.error("Failed to fetch networks: %s", err)
        raise vol.Invalid("network_error") from err


class DazeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Daze Wallbox."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise the config flow."""
        self._reauth_entry: ConfigEntry | None = None
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._email: str | None = None
        self._networks: list[dict[str, Any]] = []
        self._network_uid: str | None = None
        self._network_name: str | None = None
        self._evse_name: str | None = None
        self._serial_number: str | None = None
        self._device_profile: str | None = None
        self._firmware_version: str | None = None
        self._software_version: str | None = None
        # Set when the user signs in with an email address and a
        # password, and stored so the integration can do it again.
        self._username: str | None = None
        self._password: str | None = None
        self._reconfigure = False

    # ------------------------------------------------------------------
    # Step 1: How to sign in
    # ------------------------------------------------------------------

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step — choosing how to sign in.

        With an email address and a password the integration signs in
        by itself, now and whenever its tokens expire. Pasting the
        tokens stays available for accounts that have no password
        (Google, Apple) or need a second step to sign in.
        """
        # Show the re-auth title if this is a re-authentication flow
        if self._reauth_entry:
            self.context["title_placeholders"] = {
                "name": self._reauth_entry.title
            }

        return self.async_show_menu(
            step_id="user", menu_options=["credentials", "tokens"]
        )

    # ------------------------------------------------------------------
    # Step 1a: Email address and password
    # ------------------------------------------------------------------

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Sign in with the email address and password of the account.

        For an entry that already exists (re-authentication, or
        "Reconfigure") this is the whole flow: the tokens and the
        credentials are written to it and it is reloaded. Which charger
        it is for does not change by signing in again.
        """
        errors: dict[str, str] = {}
        entry = self._reauth_entry

        if user_input is not None:
            email = user_input[CONF_USERNAME].strip()
            password = user_input[CONF_PASSWORD]

            try:
                tokens = await _sign_in(self.hass, email, password)
                info = await _validate_tokens(
                    self.hass, tokens["access_token"], tokens["refresh_token"]
                )
            except vol.Invalid as err:
                errors["base"] = str(err)
            except Exception:
                _LOGGER.exception("Unexpected error while signing in")
                errors["base"] = "network_error"
            else:
                self._access_token = info[CONF_ACCESS_TOKEN]
                self._refresh_token = info[CONF_REFRESH_TOKEN]
                self._email = info["email"] or email
                self._username = email
                self._password = password

                if entry is None:
                    # Proceed to network selection
                    return await self.async_step_network()

                error = await self._async_check_installation(entry)
                if error is None:
                    return await self._async_update_existing(entry)
                errors["base"] = error

        default_email = ""
        if user_input is not None:
            default_email = user_input.get(CONF_USERNAME, "")
        elif entry is not None:
            default_email = entry.data.get(CONF_USERNAME) or entry.data.get(
                CONF_EMAIL, ""
            )

        return self.async_show_form(
            step_id="credentials",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_USERNAME, default=default_email): str,
                    vol.Required(CONF_PASSWORD): str,
                }
            ),
            errors=errors,
        )

    async def _async_check_installation(self, entry: ConfigEntry) -> str | None:
        """Check the account signed in to can see the entry's network.

        Signing in to a different Daze account would otherwise be
        saved, reloaded, and fail on every poll with the entry looking
        correctly set up.

        Returns:
            None if it can, otherwise the error key to show.

        """
        session = async_get_clientsession(self.hass)
        auth_client = DazeAuthClient(
            self._access_token, self._refresh_token  # type: ignore[arg-type]
        )
        api_client = DazeApiClient(auth_client, session)

        try:
            networks = await _fetch_networks(
                self.hass, api_client, self._email  # type: ignore[arg-type]
            )
        except vol.Invalid as err:
            return str(err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching networks")
            return "network_error"

        network_uid = entry.data.get(CONF_NETWORK_UID)
        if network_uid and all(
            net.get("uid") != network_uid for net in networks
        ):
            return "wrong_account"

        return None

    async def _async_update_existing(
        self, entry: ConfigEntry
    ) -> ConfigFlowResult:
        """Write the new tokens and credentials to an existing entry."""
        data = {
            **entry.data,
            CONF_ACCESS_TOKEN: self._access_token,
            CONF_REFRESH_TOKEN: self._refresh_token,
            CONF_EMAIL: self._email,
            CONF_USERNAME: self._username,
            CONF_PASSWORD: self._password,
        }
        self.hass.config_entries.async_update_entry(entry, data=data)
        await self.hass.config_entries.async_reload(entry.entry_id)

        return self.async_abort(
            reason=(
                "reconfigure_successful"
                if self._reconfigure
                else "reauth_successful"
            )
        )

    # ------------------------------------------------------------------
    # Step 1b: Token entry
    # ------------------------------------------------------------------

    async def async_step_tokens(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle token entry.

        The user provides their Daze access token and refresh token.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            # Strip pasted whitespace/newlines and quotes.
            access_token = user_input[CONF_ACCESS_TOKEN].strip().strip('"')
            refresh_token = user_input[CONF_REFRESH_TOKEN].strip().strip('"')

            try:
                info = await _validate_tokens(
                    self.hass, access_token, refresh_token
                )
            except vol.Invalid as err:
                errors["base"] = str(err)
            except Exception:
                _LOGGER.exception("Unexpected error during token validation")
                errors["base"] = "network_error"
            else:
                self._access_token = info[CONF_ACCESS_TOKEN]
                self._refresh_token = info[CONF_REFRESH_TOKEN]
                self._email = info["email"]
                # Pasted tokens say nothing about a password: one
                # stored earlier is dropped with the tokens it led to.
                self._username = None
                self._password = None

                # Proceed to network selection
                return await self.async_step_network()

        return self.async_show_form(
            step_id="tokens",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Step 2: Network selection
    # ------------------------------------------------------------------

    async def async_step_network(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the network selection step."""
        errors: dict[str, str] = {}

        # Fetch networks on first load
        if not self._networks:
            session = async_get_clientsession(self.hass)
            auth_client = DazeAuthClient(
                self._access_token, self._refresh_token  # type: ignore[arg-type]
            )
            api_client = DazeApiClient(auth_client, session)

            try:
                self._networks = await _fetch_networks(
                    self.hass, api_client, self._email  # type: ignore[arg-type]
                )
            except vol.Invalid as err:
                errors["base"] = str(err)
                return self.async_show_form(
                    step_id="network",
                    data_schema=vol.Schema({}),
                    errors=errors,
                )
            except Exception:
                _LOGGER.exception(
                    "Unexpected error fetching networks"
                )
                errors["base"] = "network_error"
                return self.async_show_form(
                    step_id="network",
                    data_schema=vol.Schema({}),
                    errors=errors,
                )

            if not self._networks:
                errors["base"] = "no_networks"
                return self.async_show_form(
                    step_id="network",
                    data_schema=vol.Schema({}),
                    errors=errors,
                )

        if user_input is not None:
            self._network_uid = user_input[CONF_NETWORK_UID]
            # Find the network name from the selected UID
            for net in self._networks:
                if net.get("uid") == self._network_uid:
                    self._network_name = net.get("name", "")
                    break

            return await self.async_step_confirm()

        # Build selector options from available networks
        network_options = {
            net["uid"]: net.get("name", "Unknown")
            for net in self._networks
        }

        network_schema = vol.Schema(
            {
                vol.Required(CONF_NETWORK_UID): vol.In(network_options),
            }
        )

        return self.async_show_form(
            step_id="network",
            data_schema=network_schema,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Step 3: Confirmation
    # ------------------------------------------------------------------

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the confirmation step — fetches EVSE info and creates the entry."""
        errors: dict[str, str] = {}

        if user_input is not None or self._serial_number:
            # Apply the user-chosen device name from the form field
            if user_input is not None and CONF_EVSE_NAME in user_input:
                self._evse_name = user_input[CONF_EVSE_NAME]

            # Create the config entry
            data = {
                CONF_ACCESS_TOKEN: self._access_token,
                CONF_REFRESH_TOKEN: self._refresh_token,
                CONF_EMAIL: self._email,
                CONF_NETWORK_UID: self._network_uid,
                CONF_NETWORK_NAME: self._network_name,
                CONF_EVSE_NAME: self._evse_name,
                CONF_SERIAL_NUMBER: self._serial_number,
                CONF_DEVICE_PROFILE: self._device_profile,
                CONF_FIRMWARE_VERSION: self._firmware_version,
                CONF_SOFTWARE_VERSION: self._software_version,
            }
            if self._username and self._password:
                data[CONF_USERNAME] = self._username
                data[CONF_PASSWORD] = self._password

            if self._reauth_entry:
                # Update existing entry (re-auth flow)
                self.hass.config_entries.async_update_entry(
                    self._reauth_entry, data=data
                )
                await self.hass.config_entries.async_reload(
                    self._reauth_entry.entry_id
                )
                return self.async_abort(reason="reauth_successful")

            # Set unique ID to prevent duplicate entries
            await self.async_set_unique_id(self._serial_number)
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=self._evse_name or "Daze Wallbox",
                data=data,
            )

        # Fetch EVSE info to populate confirmation details
        session = async_get_clientsession(self.hass)
        auth_client = DazeAuthClient(
            self._access_token, self._refresh_token  # type: ignore[arg-type]
        )
        api_client = DazeApiClient(auth_client, session)

        try:
            evses = await api_client.async_get_evses(self._network_uid)  # type: ignore[arg-type]
        except Exception:
            _LOGGER.exception("Failed to fetch EVSEs for confirmation")
            errors["base"] = "network_error"
            return self.async_show_form(
                step_id="confirm",
                data_schema=vol.Schema({}),
                errors=errors,
            )

        if not evses:
            errors["base"] = "no_evses"
            return self.async_show_form(
                step_id="confirm",
                data_schema=vol.Schema({}),
                errors=errors,
            )

        evse = evses[0]
        original_evse_name = evse.get("evseName", "Daze Wallbox")
        # Used as reported: the charger usually names itself after the
        # vendor already, so adding a suffix duplicated it.
        self._evse_name = device_name(evse)
        self._serial_number = evse.get("serialNumber", "")
        self._device_profile = evse.get("deviceProfile", "")
        self._firmware_version = evse.get("firmwareVersion", "")
        self._software_version = evse.get("softwareVersion", "")

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_EVSE_NAME, default=self._evse_name
                ): str,
            }),
            description_placeholders={
                "network_name": str(self._network_name or ""),
                "evse_name": str(original_evse_name),
                "serial_number": str(self._serial_number or ""),
                "device_profile": str(self._device_profile or ""),
            },
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Re-authentication
    # ------------------------------------------------------------------

    async def async_step_reauth(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle re-authentication when tokens are expired/invalid."""
        entry_id = self.context.get("entry_id")
        assert isinstance(entry_id, str)

        self._reauth_entry = self.hass.config_entries.async_get_entry(entry_id)
        return await self.async_step_user()

    # ------------------------------------------------------------------
    # Reconfiguration
    # ------------------------------------------------------------------

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Give an entry set up with pasted tokens a way to sign in.

        "Reconfigure" on the integration asks for the email address and
        password, without waiting for the tokens to expire first.
        """
        entry_id = self.context.get("entry_id")
        assert isinstance(entry_id, str)

        self._reauth_entry = self.hass.config_entries.async_get_entry(entry_id)
        self._reconfigure = True
        return await self.async_step_credentials()

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return DazeOptionsFlowHandler(config_entry)


class DazeOptionsFlowHandler(OptionsFlow):
    """Handle Daze Wallbox options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialise options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Let the user choose the poll interval and the grid sensor.

        Faster polling makes the entities more responsive at the cost
        of more requests against the Daze cloud API. The entry reloads
        on save, so the new interval takes effect immediately.

        The single signed grid power sensor feeds solar surplus control
        and is optional: leaving it empty is a supported configuration,
        and solar control simply refuses to arm without it.
        """
        if user_input is not None:
            # Carry forward only the one key this form does not own —
            # the solar reserve, written directly to these same options
            # by Task 7's own reserve entity — rather than blanket-
            # merging the rest of the stored options over the submitted
            # ones. Every field this form *does* own is vol.Optional
            # with no default, so clearing one in the frontend omits it
            # from user_input rather than submitting an empty value; a
            # blanket merge would read that omission as "unchanged" and
            # silently restore the stale value, making the sensor
            # impossible to clear once set — the spec calls empty a
            # supported configuration.
            data = dict(user_input)
            if CONF_SOLAR_RESERVE in self._config_entry.options:
                data[CONF_SOLAR_RESERVE] = self._config_entry.options[
                    CONF_SOLAR_RESERVE
                ]
            return self.async_create_entry(title="", data=data)

        current = self._config_entry.options.get(
            CONF_POLL_INTERVAL,
            self._config_entry.data.get(
                CONF_POLL_INTERVAL, DEFAULT_POLL_INTERVAL
            ),
        )

        options = self._config_entry.options
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_POLL_INTERVAL, default=current
                ): vol.All(
                    vol.Coerce(int),
                    vol.Range(min=MIN_POLL_INTERVAL, max=MAX_POLL_INTERVAL),
                ),
                # Optional so the integration works without solar. Solar
                # control refuses to leave "off" until the sensor is set.
                vol.Optional(
                    CONF_GRID_POWER_SENSOR,
                    description={
                        "suggested_value": options.get(CONF_GRID_POWER_SENSOR)
                    },
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor", device_class="power"
                    )
                ),
                # Optional so the form can still be saved without it,
                # not because it has a default: solar control refuses
                # to arm until it is answered.
                vol.Optional(
                    CONF_SUPPLY_PHASES,
                    description={
                        "suggested_value": options.get(CONF_SUPPLY_PHASES)
                    },
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[SUPPLY_PHASES_SINGLE, SUPPLY_PHASES_THREE],
                        translation_key="supply_phases",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
