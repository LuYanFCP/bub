"""Interactive model connection setup and discovery."""

from __future__ import annotations

import asyncio
import inspect
import os
from concurrent.futures import Future
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

import typer
from any_llm import AnyLLM
from any_llm.exceptions import AuthenticationError, MissingApiKeyError, UnsupportedProviderError

from bub import configure, inquirer
from bub.builtin.codex_provider import should_use_openai_codex_provider
from bub.builtin.settings import DEFAULT_MODEL, AgentSettings

PROVIDERS = {
    "openrouter": "OpenRouter (hosted model gateway)",
    "openai": "OpenAI (official API)",
    "openai-compatible": "OpenAI-compatible (custom URL / local server)",
    "anthropic": "Anthropic (Claude API)",
    "gemini": "Google Gemini (native API)",
    "azure": "Azure AI (Azure endpoint)",
    "bedrock": "Amazon Bedrock (AWS)",
    "ollama": "Ollama (local server)",
    "groq": "Groq",
    "mistral": "Mistral AI",
    "deepseek": "DeepSeek",
    "custom": "Other provider (SDK provider name)",
}
OPENAI_BASE = "https://api.openai.com/v1"
CONNECTION_TIMEOUT = 10
MANUAL_MODEL = "Enter a model ID manually"
EDIT_CONNECTION = "Edit URL / API key"
RETRY_CONNECTION = "Retry connection"


def _provider_value(value: str | dict[str, str] | None, provider: str, *, same_provider: bool = True) -> str:
    if isinstance(value, dict):
        return value.get(provider, "")
    return (value or "") if same_provider else ""


def _provider_class(provider: str) -> type[AnyLLM] | None:
    try:
        return AnyLLM.get_provider_class(provider)
    except (ImportError, UnsupportedProviderError):
        return None


def _default_base(provider: str) -> str:
    provider_class = _provider_class(provider)
    if provider_class is None:
        return ""
    env_name = provider_class.ENV_API_BASE_NAME
    return (
        (os.getenv(env_name) if env_name else None)
        or provider_class.API_BASE
        or ("http://localhost:11434" if provider == "ollama" else "")
    )


def _has_environment_key(provider: str) -> bool:
    if _provider_value(AgentSettings().api_key, provider):
        return True
    provider_class = _provider_class(provider)
    return bool(provider_class and any(os.getenv(name) for name in provider_class.ENV_API_KEY_NAME.split("/")))


def _saved_base(current_config: dict[str, object], provider: str, api_base: str, *, compatible: bool) -> str:
    existing = current_config.get("api_base")
    if isinstance(existing, dict):
        existing = existing.get(provider)
    if existing or compatible or api_base.rstrip("/") != _default_base(provider).rstrip("/"):
        return api_base
    return ""


def _required_text(message: str, default: str = "") -> str:
    while True:
        value = inquirer.ask_text(message, default=default).strip()
        if value:
            return value
        typer.secho("Please enter a value.", fg="yellow")


def _ask_base(default: str, *, required: bool) -> str:
    while True:
        value = inquirer.ask_text("API base URL", default=default).strip().rstrip("/")
        if not value and not required:
            return ""
        try:
            parsed = urlsplit(value)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.port != 0
            valid = valid and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
            valid = valid and not any(char.isspace() for char in value)
        except ValueError:
            valid = False
        if valid:
            return value
        typer.secho("Enter an http:// or https:// API base URL, without credentials, query or fragment.", fg="yellow")


async def _discover_models(provider: str, api_base: str, api_key: str, **client_args: Any) -> list[str]:
    async with asyncio.timeout(CONNECTION_TIMEOUT):
        llm = AnyLLM.create(provider, **{**client_args, "api_base": api_base or None, "api_key": api_key or None})
        try:
            models = await llm.alist_models()
            return sorted({model.id.strip() for model in models if isinstance(model.id, str) and model.id.strip()})
        finally:
            client = getattr(llm, "client", None)
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


def discover_models(provider: str, api_base: str, api_key: str, **client_args: Any) -> list[str]:
    """Check the models endpoint with a bounded wait, without generating tokens."""
    result: Future[list[str]] = Future()

    def run() -> None:
        try:
            result.set_result(asyncio.run(_discover_models(provider, api_base, api_key, **client_args)))
        except BaseException as exc:
            result.set_exception(exc)

    # Some SDKs (notably Bedrock) block even inside async methods, so an asyncio
    # timeout alone cannot bound the prompt's wait. A daemon also allows Ctrl+C
    # to exit while such a call finishes; its client is closed in the worker.
    Thread(target=run, name="bub-model-discovery", daemon=True).start()
    return result.result(timeout=CONNECTION_TIMEOUT)


def _connection_error(exc: Exception) -> str:
    # SDK errors may contain request URLs, response bodies or credentials.
    if isinstance(exc, (AuthenticationError, MissingApiKeyError)):
        return "Authentication failed or API key missing. Check the key and its permissions."
    if isinstance(exc, TimeoutError):
        return f"Connection timed out after {CONNECTION_TIMEOUT} seconds. Check the URL and network."
    if isinstance(exc, (NotImplementedError, UnsupportedProviderError)):
        return "Model discovery is unavailable for this provider. You can enter a model ID manually."
    if isinstance(exc, ImportError):
        return "The provider SDK is not installed. Install its dependencies or enter a model ID manually."
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "original_exception", None), "status_code", None)
    if status in {401, 403}:
        return f"Authentication rejected (HTTP {status}). Check the API key and its permissions."
    if status == 404:
        return "Models endpoint not found (HTTP 404). Check the base URL, or enter a model ID manually."
    return "Could not fetch models. Check the URL, API key and network, or enter a model ID manually."


def _choose_model(models: list[str], default: str) -> str:
    if models:
        typer.echo(f"Models endpoint reachable: found {len(models)} models.")
        typer.echo("Choose a chat model with tool support. Model calls have not been tested.")
        selected = inquirer.ask_fuzzy(
            "LLM model (type to search)",
            choices=[*models, MANUAL_MODEL],
            default=default if default in models else models[0],
        )
        if selected != MANUAL_MODEL:
            return selected
    return _required_text("LLM model", default=default)


def _connection_models(provider: str, api_base: str, api_key: str, **client_args: Any) -> list[str] | None:
    """Return models, or None when the user wants to edit the connection."""
    while True:
        typer.echo(f"Checking connection and fetching models (up to {CONNECTION_TIMEOUT}s)...")
        try:
            models = discover_models(provider, api_base, api_key, **client_args)
        except Exception as exc:
            typer.secho(_connection_error(exc), fg="yellow")
        else:
            if not models:
                typer.echo("Models endpoint reachable, but no models were returned. Enter a model ID manually.")
            return models
        action = inquirer.ask_select(
            "Connection check failed",
            choices=[EDIT_CONNECTION, RETRY_CONNECTION, MANUAL_MODEL],
            default=EDIT_CONNECTION,
        )
        if action == EDIT_CONNECTION:
            return None
        if action == MANUAL_MODEL:
            return []


def _connection_config(
    current_config: dict[str, object], provider: str, api_base: str, api_key: str
) -> dict[str, object]:
    config: dict[str, object] = {}
    for name, value in (("api_base", api_base), ("api_key", api_key)):
        existing = current_config.get(name)
        if isinstance(existing, dict):
            # A newly inserted empty value would mask provider-specific env vars.
            if value or provider in existing:
                config[name] = {provider: value}
        elif value or name in current_config:
            config[name] = value or None
    return config


def _check_connection(
    current_config: dict[str, object], provider: str, api_base: str, api_key: str
) -> list[str] | None:
    config = configure.merge({}, current_config, _connection_config(current_config, provider, api_base, api_key))
    settings = AgentSettings.model_validate(config)
    effective_base = _provider_value(settings.api_base, provider)
    effective_key = _provider_value(settings.api_key, provider)
    if (effective_base, effective_key) != (api_base, api_key):
        typer.echo("BUB_* environment settings override this connection's URL or API key.")
    if should_use_openai_codex_provider(provider, "", api_key=effective_key or None, api_base=effective_base or None):
        typer.echo("Using OpenAI OAuth login. Model discovery is unavailable; enter a model ID manually.")
        return []
    return _connection_models(
        provider,
        **{**settings.client_args, "api_base": effective_base or _default_base(provider), "api_key": effective_key},
    )


def _select_provider(current_choice: str) -> tuple[str, bool]:
    choices = dict(PROVIDERS)
    if current_choice not in choices:
        choices[current_choice] = current_choice
    selected = inquirer.ask_fuzzy("LLM provider", choices=list(choices.values()), default=choices[current_choice])
    selection = next((name for name, label in choices.items() if label == selected), selected)
    custom = selection == "custom"
    if custom:
        selection = _required_text("Custom provider")
    return selection, custom


def _current_service(settings: AgentSettings) -> tuple[str, str, str]:
    current_provider, separator, current_model = settings.model.partition(":")
    if not separator:
        current_provider, _, fallback_model = DEFAULT_MODEL.partition(":")
        current_model = settings.model.strip() or fallback_model
    current_base = _provider_value(settings.api_base, current_provider) or _default_base(current_provider)
    current_choice = current_provider
    if current_provider == "openai" and current_base.rstrip("/") != OPENAI_BASE:
        current_choice = "openai-compatible"
    return current_choice, current_base, current_model


def collect_model_config(current_config: dict[str, object]) -> dict[str, object]:
    settings = AgentSettings.model_validate(current_config)
    current_choice, current_base, current_model = _current_service(settings)
    selection, custom = _select_provider(current_choice)
    compatible = selection == "openai-compatible"
    provider = "openai" if compatible else selection
    same_service = selection == current_choice
    api_base = _provider_value(settings.api_base, provider, same_provider=same_service)
    if not api_base:
        api_base = "" if compatible else _default_base(provider)
    if provider == "openai" and not compatible:
        api_base = OPENAI_BASE
    saved_key = current_config.get("api_key")
    api_key = _provider_value(
        saved_key if isinstance(saved_key, (str, dict)) else None, provider, same_provider=same_service
    )
    current_provider = "openai" if current_choice == "openai-compatible" else current_choice
    key_base = current_base if provider == current_provider else api_base
    model_default = current_model if same_service else ""

    ask_base = compatible or custom or provider in {"azure", "ollama"} or api_base != _default_base(provider)
    if api_base:
        typer.echo(f"API endpoint: {api_base}")
    if compatible:
        typer.echo(
            "Use your server's URL and API key. A blank key uses saved or environment credentials when available."
        )
    else:
        typer.echo("Leave the API key blank to use the provider's environment credentials.")
    while True:
        if ask_base:
            typer.echo("Enter the API base URL, including /v1 if required; omit /chat/completions or /models.")
            api_base = _ask_base(api_base, required=compatible or provider == "azure")
        if api_base != key_base.rstrip("/"):
            api_key = ""
            model_default = ""
        key_prompt = "API key (Enter to keep current key)" if api_key else "API key (optional)"
        api_key = inquirer.ask_secret(key_prompt).strip() or api_key
        if compatible and not api_key and not _has_environment_key(provider):
            # The OpenAI SDK requires a nonempty key even for servers without auth.
            api_key = "not-required"
        key_base = api_base
        # Leave SDK defaults implicit, preserving environment and OAuth resolution at runtime.
        saved_base = _saved_base(current_config, provider, api_base, compatible=compatible)
        models = _check_connection(current_config, provider, saved_base, api_key)
        if models is not None:
            break
        ask_base = True

    model = _choose_model(models, model_default)
    config = _connection_config(current_config, provider, saved_base, api_key)
    config["model"] = f"{provider}:{model}"
    return config
