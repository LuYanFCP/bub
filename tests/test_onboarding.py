from __future__ import annotations

import asyncio
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import typer
from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import AuthenticationError
from typer.testing import CliRunner

from bub import configure, inquirer
from bub.builtin import codex_provider, onboarding
from bub.framework import BubFramework
from bub.hooks import hookimpl


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(codex_provider, "load_openai_codex_oauth_tokens", lambda: None)
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def prompts(monkeypatch):
    calls: list[tuple[str, object]] = []
    answers = {"provider": "openai", "base": "https://example.test/v1", "key": "test-key", "model": "model-b"}

    def fuzzy(message, choices, default=None):
        calls.append((message, default))
        if message == "LLM provider":
            label = onboarding.PROVIDERS[answers["provider"]]
            assert label in choices
            return label
        assert "model-a" in choices
        assert onboarding.MANUAL_MODEL in choices
        return answers["model"]

    def text(message, default=""):
        calls.append((message, default))
        return answers["base"] if message == "API base URL" else answers["model"]

    def secret(message):
        calls.append((message, None))
        return answers["key"]

    def discover(provider, base, key):
        calls.append(("discover", (provider, base, key)))
        return ["model-a", "model-b"]

    monkeypatch.setattr(inquirer, "ask_fuzzy", fuzzy)
    monkeypatch.setattr(inquirer, "ask_text", text)
    monkeypatch.setattr(inquirer, "ask_secret", secret)
    monkeypatch.setattr(inquirer, "ask_select", lambda *args, **kwargs: onboarding.MANUAL_MODEL)
    monkeypatch.setattr(inquirer, "ask_checkbox", lambda *args, **kwargs: [])
    monkeypatch.setattr(inquirer, "ask_confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(onboarding, "discover_models", discover)
    return answers, calls


@pytest.mark.parametrize("provider", [p for p in onboarding.PROVIDERS if p != "custom"])
def test_all_providers_check_connection_before_selecting_models(prompts, provider):
    answers, calls = prompts
    answers["provider"] = provider
    config = onboarding.collect_model_config({})

    names = [name for name, _ in calls]
    assert names[0] == "LLM provider"
    assert names.index("API key (optional)") < names.index("discover") < names.index("LLM model (type to search)")
    if provider in {"openai-compatible", "azure", "ollama"}:
        assert names.index("API base URL") < names.index("API key (optional)")
    else:
        assert "API base URL" not in names
    saved_provider = "openai" if provider == "openai-compatible" else provider
    assert config["model"] == f"{saved_provider}:model-b"
    assert config["api_key"] == "test-key"
    if provider == "openai-compatible":
        assert config["api_base"] == "https://example.test/v1"
        assert dict(calls)["discover"] == ("openai", "https://example.test/v1", "test-key")


def test_changed_provider_does_not_reuse_model_or_credentials(prompts, monkeypatch):
    _, calls = prompts
    monkeypatch.setattr(onboarding, "discover_models", lambda *args: [])
    config = onboarding.collect_model_config({
        "model": "openrouter:openrouter/free",
        "api_key": "old-key",
        "api_base": "https://old.test/v1",
    })

    assert dict(calls)["LLM model"] == ""
    assert config["model"] == "openai:model-b"
    assert config["api_base"] == onboarding.OPENAI_BASE
    assert "API key (Enter to keep current key)" not in dict(calls)


def test_failed_connection_can_edit_url_and_key_then_select_models(prompts, monkeypatch, capsys):
    answers, calls = prompts
    answers["provider"] = "openai-compatible"
    urls = iter(["https://wrong.test/v1", "https://correct.test/v1"])
    keys = iter(["bad-key", "good-key"])
    probes = []
    monkeypatch.setattr(inquirer, "ask_text", lambda *args, **kwargs: next(urls))
    monkeypatch.setattr(inquirer, "ask_secret", lambda *args: next(keys))
    monkeypatch.setattr(inquirer, "ask_select", lambda *args, **kwargs: onboarding.EDIT_CONNECTION)

    def discover(provider, base, key):
        probes.append((provider, base, key))
        if len(probes) == 1:
            raise AuthenticationError("response body containing bad-key")
        return ["model-a", "model-b"]

    monkeypatch.setattr(onboarding, "discover_models", discover)
    config = onboarding.collect_model_config({})

    assert probes == [("openai", "https://wrong.test/v1", "bad-key"), ("openai", "https://correct.test/v1", "good-key")]
    assert config == {"model": "openai:model-b", "api_base": "https://correct.test/v1", "api_key": "good-key"}
    assert [name for name, _ in calls].count("LLM provider") == 1
    assert "bad-key" not in capsys.readouterr().out


def test_connection_can_retry_without_reentering_credentials(prompts, monkeypatch):
    _, calls = prompts
    attempts = []

    def discover(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise TimeoutError
        return ["model-a", "model-b"]

    monkeypatch.setattr(onboarding, "discover_models", discover)
    monkeypatch.setattr(inquirer, "ask_select", lambda *args, **kwargs: onboarding.RETRY_CONNECTION)
    onboarding.collect_model_config({})
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert [name for name, _ in calls].count("API key (optional)") == 1


@pytest.mark.parametrize("failure", [TimeoutError(), NotImplementedError(), ImportError(), ValueError("sensitive key")])
def test_failed_discovery_allows_manual_model_entry(prompts, monkeypatch, failure, capsys):
    _, calls = prompts

    def discover(*args):
        raise failure

    monkeypatch.setattr(onboarding, "discover_models", discover)
    config = onboarding.collect_model_config({})
    assert config["model"] == "openai:model-b"
    assert "LLM model (type to search)" not in dict(calls)
    assert "sensitive key" not in capsys.readouterr().out


def test_successful_discovery_still_allows_unlisted_model(prompts, monkeypatch):
    answers, _ = prompts
    answers["model"] = onboarding.MANUAL_MODEL
    monkeypatch.setattr(inquirer, "ask_text", lambda *args, **kwargs: "private-model")
    assert onboarding.collect_model_config({})["model"] == "openai:private-model"


def test_compatible_server_without_auth_gets_a_working_saved_key(prompts):
    answers, calls = prompts
    answers.update(provider="openai-compatible", key="")
    config = onboarding.collect_model_config({})
    assert config["api_key"] == "not-required"
    assert dict(calls)["discover"] == ("openai", "https://example.test/v1", "not-required")


def test_environment_key_is_used_for_check_but_not_written_to_config(prompts, monkeypatch):
    answers, calls = prompts
    answers["key"] = ""
    monkeypatch.setenv("BUB_API_KEY", "environment-key")
    config = onboarding.collect_model_config({})
    assert dict(calls)["discover"] == ("openai", onboarding.OPENAI_BASE, "environment-key")
    assert "api_key" not in config
    assert "api_base" not in config


@pytest.mark.parametrize("variable", ["BUB_API_BASE", "BUB_OPENAI_API_BASE"])
def test_connection_check_honors_environment_overrides_and_client_options(prompts, monkeypatch, capsys, variable):
    monkeypatch.setenv(variable, "https://override.test/v1")
    received = []

    def discover(*args, **kwargs):
        received.append((args, kwargs))
        return ["model-a", "model-b"]

    monkeypatch.setattr(onboarding, "discover_models", discover)
    onboarding.collect_model_config({"client_args": {"default_headers": {"X-Workspace": "test"}}})
    assert received == [
        (("openai", "https://override.test/v1", "test-key"), {"default_headers": {"X-Workspace": "test"}})
    ]
    assert "environment settings override" in capsys.readouterr().out


def test_connection_check_overrides_auth_fields_in_client_args_like_runtime(prompts, monkeypatch):
    received = []

    def discover(*args, **kwargs):
        received.append((args, kwargs))
        return ["model-a", "model-b"]

    monkeypatch.setattr(onboarding, "discover_models", discover)
    onboarding.collect_model_config({
        "client_args": {
            "api_key": "unused-key",
            "api_base": "https://unused.test/v1",
            "timeout": 5,
        }
    })
    assert received == [(("openai", onboarding.OPENAI_BASE, "test-key"), {"timeout": 5})]


@pytest.mark.parametrize("compatible", [False, True])
def test_other_provider_config_does_not_shadow_environment_connection(prompts, monkeypatch, compatible):
    answers, calls = prompts
    answers.update(provider="openai-compatible" if compatible else "openai", key="")
    monkeypatch.setenv("BUB_OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("BUB_OPENAI_API_BASE", "https://example.test/v1")
    current = {
        "api_key": {"anthropic": "other-key"},
        "api_base": {"anthropic": "https://other.test"},
    }
    update = onboarding.collect_model_config(current)
    assert dict(calls)["discover"] == ("openai", "https://example.test/v1", "environment-key")
    assert "api_key" not in update
    merged = configure.merge({}, current, update)
    settings = onboarding.AgentSettings.model_validate(merged)
    assert settings.model_client_kwargs(LLMProvider.OPENAI) == {
        "api_base": "https://example.test/v1",
        "api_key": "environment-key",
    }
    assert merged["api_key"] == {"anthropic": "other-key"}


def test_provider_credentials_keep_other_providers_when_merged(prompts):
    current = {
        "model": "anthropic:old-model",
        "api_key": {"anthropic": "keep-key"},
        "api_base": {"anthropic": "https://old.test"},
    }
    merged = configure.merge({}, current, onboarding.collect_model_config(current))
    assert merged["api_key"] == {"anthropic": "keep-key", "openai": "test-key"}
    assert merged["api_base"] == {"anthropic": "https://old.test"}


def test_switching_from_compatible_to_official_does_not_reuse_compatible_key(prompts):
    answers, calls = prompts
    answers["key"] = ""
    config = onboarding.collect_model_config({
        "model": "openai:private-model",
        "api_key": {"openai": "private-key"},
        "api_base": {"openai": "https://private.test/v1"},
    })
    assert dict(calls)["discover"] == ("openai", onboarding.OPENAI_BASE, "")
    assert config["api_key"] == {"openai": ""}


def test_onboarding_hooks_only_receive_current_run_contributions(prompts, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUB_HOME", str(tmp_path / "home"))
    config_file = tmp_path / "config.yml"
    original = {
        "model": "openai:model-b",
        "api_key": "keep-key",
        "api_base": onboarding.OPENAI_BASE,
        "max_tokens": 8192,
        "enabled_channels": "",
        "telegram": {"token": "keep-token"},
    }
    configure.save(config_file, original)
    framework = BubFramework(config_file=config_file)
    framework.load_hooks()
    received = []

    class Observer:
        @hookimpl
        def onboard_config(self, current_config):
            received.append(dict(current_config))
            return {"plugin": {"configured": True}}

    framework.plugin_manager.register(Observer(), name="observer")
    result = CliRunner().invoke(framework.create_cli_app(), ["onboard"])

    assert result.exit_code == 0, result.output
    builtin_config = {"model": "openai:model-b", "api_key": "test-key", "enabled_channels": "", "stream_output": False}
    assert received == [builtin_config]
    assert configure.load(config_file) == {**builtin_config, "plugin": {"configured": True}}
    assert "keep-key" not in result.output


@pytest.mark.parametrize("variable", ["OPENAI_API_KEY", "BUB_API_KEY", "BUB_OPENAI_API_KEY"])
def test_blank_compatible_key_uses_environment_for_discovery_and_saved_config(prompts, monkeypatch, variable):
    answers, _ = prompts
    answers.update(provider="openai-compatible", key="")
    monkeypatch.setenv(variable, "environment-key")
    real_discover = onboarding._discover_models
    monkeypatch.setattr(
        onboarding, "discover_models", lambda *args, **kwargs: asyncio.run(real_discover(*args, **kwargs))
    )
    real_create = AnyLLM.create
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "test"}
                    for model in ["model-a", "model-b"]
                ],
            },
        )

    def create_client(provider, **kwargs):
        return real_create(provider, **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(AnyLLM, "create", create_client)
    config = onboarding.collect_model_config({})
    assert "api_key" not in config
    runtime_settings = onboarding.AgentSettings.model_validate(config)
    runtime_key = runtime_settings._provider_value(runtime_settings.api_key, LLMProvider.OPENAI)
    onboarding.discover_models("openai", config["api_base"], runtime_key or "")
    assert len(requests) == 2
    assert all(request.headers["authorization"] == "Bearer environment-key" for request in requests)
    assert all(str(request.url) == "https://example.test/v1/models" for request in requests)


def test_compatible_key_detects_provider_mapping_in_environment(prompts, monkeypatch):
    answers, calls = prompts
    answers.update(provider="openai-compatible", key="")
    monkeypatch.setenv("BUB_API_KEY", '{"openai": "environment-key"}')
    config = onboarding.collect_model_config({})
    assert "api_key" not in config
    assert dict(calls)["discover"] == ("openai", "https://example.test/v1", "environment-key")


def test_oauth_login_skips_api_key_discovery_and_preserves_runtime_auth(prompts, monkeypatch, capsys):
    answers, calls = prompts
    answers["key"] = ""
    monkeypatch.setattr(codex_provider, "load_openai_codex_oauth_tokens", lambda: object())
    config = onboarding.collect_model_config({})
    assert config == {"model": "openai:model-b"}
    assert "discover" not in dict(calls)
    assert "OAuth login" in capsys.readouterr().out
    assert codex_provider.should_use_openai_codex_provider("openai", "model-b", api_key=None, api_base=None)


@pytest.mark.parametrize("initial", [{}, {"api_base": None}, {"api_base": ""}, {"api_base": {"openai": ""}}])
def test_editing_unchanged_default_url_does_not_disable_oauth(prompts, monkeypatch, initial):
    answers, _ = prompts
    answers.update(key="", base=onboarding.OPENAI_BASE)
    actions = iter([onboarding.EDIT_CONNECTION, onboarding.MANUAL_MODEL])
    monkeypatch.setattr(inquirer, "ask_select", lambda *args, **kwargs: next(actions))

    def fail(*args, **kwargs):
        raise AuthenticationError("no API key")

    monkeypatch.setattr(onboarding, "discover_models", fail)
    config = onboarding.collect_model_config(initial)
    settings = onboarding.AgentSettings.model_validate(config)
    api_base = onboarding._provider_value(settings.api_base, "openai")
    assert not api_base
    monkeypatch.setattr(codex_provider, "load_openai_codex_oauth_tokens", lambda: object())
    assert codex_provider.should_use_openai_codex_provider("openai", "model-b", api_key=None, api_base=api_base)


def test_explicit_official_base_does_not_switch_to_oauth(prompts, monkeypatch):
    answers, calls = prompts
    answers["key"] = ""
    monkeypatch.setattr(codex_provider, "load_openai_codex_oauth_tokens", lambda: object())
    config = onboarding.collect_model_config({"model": "openai:model-b", "api_base": onboarding.OPENAI_BASE})
    assert config["api_base"] == onboarding.OPENAI_BASE
    assert "discover" in dict(calls)


def test_cancelled_connection_setup_leaves_config_untouched(prompts, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUB_HOME", str(tmp_path / "home"))
    config_file = tmp_path / "config.yml"
    original = "model: openai:model-b\napi_key: keep-key\n"
    config_file.write_text(original)
    framework = BubFramework(config_file=config_file)
    framework.load_hooks()

    def cancel(*args):
        raise typer.Abort()

    monkeypatch.setattr(inquirer, "ask_secret", cancel)
    result = CliRunner().invoke(framework.create_cli_app(), ["onboard"])
    assert result.exit_code == 1
    assert config_file.read_text() == original


@pytest.mark.parametrize("invalid", ["", "example.test", "ftp://example.test", "https://host:bad", "https://host/a b"])
def test_compatible_url_is_validated_before_sending_credentials(monkeypatch, invalid):
    entries = iter([invalid, "https://example.test/custom/v1/"])
    monkeypatch.setattr(inquirer, "ask_text", lambda *args, **kwargs: next(entries))
    assert onboarding._ask_base("", required=True) == "https://example.test/custom/v1"


def test_discovery_uses_entered_endpoint_and_credentials_and_closes_client(monkeypatch):
    requests = []
    clients = []
    create = AnyLLM.create

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "created": 0, "owned_by": "test"}
                    for name in ["model-b", "model-a", "model-a"]
                ],
            },
        )

    def create_client(provider, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        return create(provider, **kwargs, http_client=client)

    monkeypatch.setattr(AnyLLM, "create", create_client)
    assert onboarding.discover_models("openai", "https://example.test/custom/v1", "test-key") == ["model-a", "model-b"]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.test/custom/v1/models"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert requests[0].method == "GET"
    assert clients[0].is_closed


def test_discovery_timeout_cancels_request_and_closes_client(monkeypatch):
    closed = AsyncMock()
    cancelled = []

    async def list_models():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(onboarding, "CONNECTION_TIMEOUT", 0.01)
    monkeypatch.setattr(
        AnyLLM,
        "create",
        lambda *args, **kwargs: SimpleNamespace(alist_models=list_models, client=SimpleNamespace(close=closed)),
    )
    with pytest.raises(TimeoutError):
        asyncio.run(onboarding._discover_models("openai", "https://example.test/v1", "test-key"))
    assert cancelled == [True]
    closed.assert_awaited_once()


@pytest.mark.parametrize("blocking_stage", ["create", "list_models"])
def test_discovery_timeout_also_bounds_blocking_provider_sdks(monkeypatch, blocking_stage):
    release = Event()
    started = Event()
    closed = Event()

    def block():
        started.set()
        release.wait(1)

    async def list_models():
        if blocking_stage == "list_models":
            block()
        return []

    def create(*args, **kwargs):
        if blocking_stage == "create":
            block()
        return SimpleNamespace(alist_models=list_models, client=SimpleNamespace(close=closed.set))

    monkeypatch.setattr(onboarding, "CONNECTION_TIMEOUT", 0.01)
    monkeypatch.setattr(AnyLLM, "create", create)
    try:
        with pytest.raises(TimeoutError):
            onboarding.discover_models("bedrock", "", "")
        assert started.wait(1)
    finally:
        release.set()
        assert closed.wait(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,provider", [("", "openrouter"), ("official", "openai"), ("compatible", "openai-compatible")]
)
async def test_real_provider_picker_distinguishes_official_and_compatible_services(query, provider):
    from inquirer_textual.InquirerApp import InquirerApp
    from inquirer_textual.widgets.InquirerFuzzy import InquirerFuzzy

    app = InquirerApp()
    app.widget = InquirerFuzzy(
        "LLM provider", list(onboarding.PROVIDERS.values()), default=onboarding.PROVIDERS["openrouter"]
    )
    async with app.run_test() as pilot:
        if query:
            await pilot.press(*query)
        await pilot.press("enter")
        await pilot.pause()
        assert inquirer.ask_prompt(app.return_value) == onboarding.PROVIDERS[provider]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys,expected",
    [(["down", "enter"], onboarding.RETRY_CONNECTION), (["down", "down", "enter"], onboarding.MANUAL_MODEL)],
)
async def test_real_connection_failure_picker_supports_keyboard_navigation(keys, expected):
    from inquirer_textual.InquirerApp import InquirerApp
    from inquirer_textual.widgets.InquirerSelect import InquirerSelect

    app = InquirerApp()
    app.widget = InquirerSelect(
        "Connection check failed",
        [onboarding.EDIT_CONNECTION, onboarding.RETRY_CONNECTION, onboarding.MANUAL_MODEL],
        default=onboarding.EDIT_CONNECTION,
    )
    async with app.run_test() as pilot:
        await pilot.press(*keys)
        await pilot.pause()
        assert inquirer.ask_prompt(app.return_value) == expected


@pytest.mark.asyncio
async def test_real_model_picker_can_search_and_select_a_discovered_model():
    from inquirer_textual.InquirerApp import InquirerApp
    from inquirer_textual.widgets.InquirerFuzzy import InquirerFuzzy

    app = InquirerApp()
    app.widget = InquirerFuzzy(
        "LLM model (type to search)", ["model-a", "model-b", onboarding.MANUAL_MODEL], mandatory=False
    )
    async with app.run_test() as pilot:
        await pilot.press(*"model-b")
        await pilot.press("enter")
        await pilot.pause()
        assert inquirer.ask_prompt(app.return_value) == "model-b"
