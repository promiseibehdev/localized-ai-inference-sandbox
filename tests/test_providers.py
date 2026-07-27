from __future__ import annotations

import pytest

from src.providers import (
    DEFAULT_PROVIDER,
    DemoProvider,
    MAX_PROMPT_CHARACTERS,
    ProviderConfigurationError,
    ProviderFactory,
    SIMULATION_DISCLOSURE,
    create_provider,
    resolve_provider_name,
)


def test_demo_is_the_safe_default_without_configuration():
    provider = create_provider(environment={})

    assert isinstance(provider, DemoProvider)
    assert DEFAULT_PROVIDER == "demo"


def test_demo_readiness_is_clear_and_requires_no_external_service():
    readiness = DemoProvider().get_readiness()

    assert readiness.ready is True
    assert readiness.simulated is True
    assert readiness.mode == "SIMULATED"
    assert "no AI model or external API" in readiness.message
    assert readiness.available_models == ["simulated-portfolio-assistant"]


def test_demo_response_is_realistic_but_unambiguously_simulated():
    result = DemoProvider().generate("Explain local AI inference in simple terms.")

    assert result.error is None
    assert result.simulated is True
    assert result.mode == "SIMULATED"
    assert "SIMULATED DEMO RESPONSE" in result.text
    assert "No real AI model was used" in result.text
    assert "Local inference runs a model" in result.text
    assert result.elapsed_seconds >= 0


def test_demo_response_handles_general_prompts_without_claiming_inference():
    result = DemoProvider().generate("What could this portfolio demonstrate?")

    assert "limited built-in knowledge base" in result.text
    assert "real configured AI provider" in result.text
    assert result.provider == "Built-in Demo"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("What is Docker?", "packages an application and its dependencies"),
        ("Explain cloud computing", "provides servers, storage, databases"),
    ],
)
def test_demo_answers_reviewed_common_topics(prompt, expected):
    result = DemoProvider().generate(prompt)

    assert expected in result.text
    assert result.error is None


def test_demo_supports_a_conservative_spelling_variation():
    result = DemoProvider().generate("Tell me about Dockre")

    assert "Docker packages an application" in result.text


def test_demo_uses_honest_fallback_for_unknown_topics():
    result = DemoProvider().generate("What is quantum biology?")

    assert "limited built-in knowledge base" in result.text
    assert "does not have a prepared answer" in result.text


@pytest.mark.parametrize(
    "prompt",
    [
        "What is Docker?",
        "Explain cloud computing",
        "What is an unsupported subject?",
    ],
)
def test_every_demo_answer_has_the_exact_simulation_disclosure(prompt):
    result = DemoProvider().generate(prompt)

    assert result.text.startswith(SIMULATION_DISCLOSURE)


def test_demo_rejects_empty_prompts_cleanly():
    result = DemoProvider().generate("   ")

    assert result.text == ""
    assert result.error == "A non-empty prompt is required."
    assert result.simulated is True


def test_demo_rejects_oversized_prompts_cleanly():
    result = DemoProvider().generate("x" * (MAX_PROMPT_CHARACTERS + 1))

    assert result.text == ""
    assert result.error == "Prompts are limited to 2,000 characters."
    assert result.simulated is True


def test_provider_resolution_precedence_is_explicit_then_settings_then_environment():
    settings = {"INFERENCE_PROVIDER": "settings-provider"}
    environment = {"INFERENCE_PROVIDER": "environment-provider"}

    assert resolve_provider_name("Explicit-Provider", settings, environment) == "explicit-provider"
    assert resolve_provider_name(None, settings, environment) == "settings-provider"
    assert resolve_provider_name(None, {}, environment) == "environment-provider"
    assert resolve_provider_name(None, {}, {}) == "demo"


def test_unknown_provider_fails_instead_of_silently_misrepresenting_output():
    with pytest.raises(ProviderConfigurationError, match="not configured"):
        create_provider("paid-or-unknown-provider", environment={})


def test_factory_supports_future_providers_without_ui_changes():
    factory = ProviderFactory()
    factory.register("test-provider", lambda _settings: DemoProvider())

    provider = factory.create("test-provider", environment={})

    assert isinstance(provider, DemoProvider)


def test_factory_keeps_local_ollama_available_only_when_selected():
    provider = create_provider(
        "ollama",
        settings={
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_MODEL": "phi3",
            "OLLAMA_HEALTH_TIMEOUT": 1.0,
            "OLLAMA_GENERATION_TIMEOUT": 90.0,
        },
        environment={},
    )

    assert provider.provider_name == "Local Ollama"
    assert provider.client.base_url == "http://127.0.0.1:11434"
    assert provider.selected_model == "phi3"
    assert provider.client.health_timeout_seconds == 1.0
    assert provider.client.generation_timeout_seconds == 90.0


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("OLLAMA_HEALTH_TIMEOUT", 0),
        ("OLLAMA_HEALTH_TIMEOUT", "not-a-number"),
        ("OLLAMA_GENERATION_TIMEOUT", -1),
        ("OLLAMA_GENERATION_TIMEOUT", float("inf")),
    ],
)
def test_ollama_rejects_invalid_timeout_configuration(setting, value):
    with pytest.raises(ProviderConfigurationError):
        create_provider(
            "ollama",
            settings={setting: value},
            environment={},
        )


@pytest.mark.parametrize(
    "url",
    [
        "localhost:11434",
        "file:///tmp/ollama.sock",
        "ftp://localhost:11434",
        "http://user:password@localhost:11434",
    ],
)
def test_ollama_rejects_unsafe_or_invalid_urls(url):
    with pytest.raises(ProviderConfigurationError):
        create_provider(
            "ollama",
            settings={"OLLAMA_BASE_URL": url},
            environment={},
        )
