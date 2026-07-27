from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isfinite
import os
import re
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol
from urllib.parse import urlparse


DEFAULT_PROVIDER = "demo"
PROVIDER_ENV_VAR = "INFERENCE_PROVIDER"
MAX_PROMPT_CHARACTERS = 2_000
DEMO_PROVIDER_VERSION = 2
SIMULATION_DISCLOSURE = "SIMULATED DEMO RESPONSE - No real AI model was used."

DEMO_KNOWLEDGE_BASE = {
    "cloud computing": (
        "Cloud computing provides servers, storage, databases, and other "
        "technology services over the internet. Organizations can use these "
        "resources when needed instead of owning all the physical hardware themselves."
    ),
    "docker": (
        "Docker packages an application and its dependencies into a container. "
        "This helps the application run consistently across development, testing, "
        "and production environments."
    ),
    "kubernetes": (
        "Kubernetes manages groups of containers across multiple computers. It "
        "helps deploy, scale, restart, and organize containerized applications."
    ),
    "python": (
        "Python is a popular programming language known for readable syntax. It "
        "is widely used for automation, web development, data analysis, and AI."
    ),
    "aws": (
        "AWS, or Amazon Web Services, is a cloud platform that provides computing, "
        "storage, databases, networking, and many other online technology services."
    ),
    "linux": (
        "Linux is an open-source operating system used on servers, computers, "
        "cloud platforms, and embedded devices. It is valued for its flexibility "
        "and reliability."
    ),
    "git": (
        "Git is a version-control system that records changes to files. It helps "
        "developers work safely, compare revisions, and collaborate on code."
    ),
    "github": (
        "GitHub is an online platform for hosting Git repositories. It adds tools "
        "for collaboration, code review, issue tracking, and automated workflows."
    ),
    "terraform": (
        "Terraform is an infrastructure-as-code tool. It lets teams describe cloud "
        "and infrastructure resources in files, then create and update them consistently."
    ),
    "devops": (
        "DevOps is a way of combining software development and IT operations. It "
        "uses collaboration and automation to build, test, and release software reliably."
    ),
    "cybersecurity": (
        "Cybersecurity protects computers, networks, applications, and data from "
        "unauthorized access, damage, or disruption. It combines technology, safe "
        "processes, and user awareness."
    ),
    "networking": (
        "Computer networking connects devices so they can exchange data and share "
        "resources. Networks use agreed rules, addresses, and equipment such as "
        "routers and switches."
    ),
    "artificial intelligence": (
        "Artificial intelligence is the broad field of building computer systems "
        "that perform tasks associated with human intelligence, such as recognizing "
        "patterns, understanding language, and making decisions."
    ),
    "machine learning": (
        "Machine learning is a branch of artificial intelligence in which systems "
        "learn patterns from data to make predictions or decisions without being "
        "given a separate rule for every situation."
    ),
    "streamlit": (
        "Streamlit is an open-source Python framework for building interactive data "
        "and AI web applications. Developers can create a user interface directly "
        "from Python code."
    ),
    "ollama": (
        "Ollama is a tool for downloading and running supported language models on "
        "a local computer. It can keep inference local, subject to the computer's "
        "available memory and processing power."
    ),
    "wordpress": (
        "WordPress is an open-source content management system for creating and "
        "managing websites. Themes control presentation, while plugins add features."
    ),
}

DEMO_TOPIC_ALIASES = {
    "machine learning": ("machine learning", "ml"),
    "artificial intelligence": ("artificial intelligence", "ai"),
    "cloud computing": ("cloud computing", "the cloud", "cloud"),
    "cybersecurity": ("cybersecurity", "cyber security", "information security"),
    "kubernetes": ("kubernetes", "k8s"),
    "github": ("github",),
    "terraform": ("terraform",),
    "networking": ("computer networking", "networking", "network"),
    "streamlit": ("streamlit",),
    "wordpress": ("wordpress",),
    "docker": ("docker",),
    "python": ("python",),
    "linux": ("linux",),
    "devops": ("devops", "dev ops"),
    "ollama": ("ollama",),
    "aws": ("amazon web services", "aws"),
    "git": ("git",),
}

UNKNOWN_DEMO_TOPIC_RESPONSE = (
    "This public demo uses a limited built-in knowledge base and does not have "
    "a prepared answer for that topic. A real configured AI provider could "
    "generate a more complete response."
)


@dataclass(frozen=True)
class ProviderReadiness:
    """Provider-neutral status information for the Streamlit interface."""

    provider: str
    mode: str
    ready: bool
    simulated: bool
    message: str
    available_models: List[str]
    selected_model: str


@dataclass(frozen=True)
class GenerationResult:
    """Provider-neutral response returned by every inference implementation."""

    text: str
    provider: str
    mode: str
    model: str
    simulated: bool
    elapsed_seconds: float
    error: Optional[str] = None


class InferenceProvider(Protocol):
    """Contract implemented by built-in, local, and future cloud providers."""

    provider_name: str
    selected_model: str

    def list_models(self) -> List[str]:
        ...

    def get_readiness(self) -> ProviderReadiness:
        ...

    def generate(self, prompt: str, model_name: Optional[str] = None) -> GenerationResult:
        ...


class ProviderConfigurationError(ValueError):
    """Raised when an unavailable provider is explicitly requested."""


class DemoProvider:
    """Permanent offline-safe provider for the public portfolio deployment."""

    provider_name = "Built-in Demo"
    model_name = "simulated-portfolio-assistant"

    def __init__(self) -> None:
        self.selected_model = self.model_name

    def list_models(self) -> List[str]:
        return [self.model_name]

    def get_readiness(self) -> ProviderReadiness:
        return ProviderReadiness(
            provider=self.provider_name,
            mode="SIMULATED",
            ready=True,
            simulated=True,
            message=(
                "Built-in Demo Mode is ready. Responses are simulated locally "
                "by the application; no AI model or external API is being used. "
                f"Demo Provider Version: {DEMO_PROVIDER_VERSION}"
            ),
            available_models=self.list_models(),
            selected_model=self.selected_model,
        )

    def generate(self, prompt: str, model_name: Optional[str] = None) -> GenerationResult:
        started_at = perf_counter()
        normalized_prompt = prompt.strip()
        selected_model = model_name or self.selected_model

        if not normalized_prompt:
            return GenerationResult(
                text="",
                provider=self.provider_name,
                mode="SIMULATED",
                model=selected_model,
                simulated=True,
                elapsed_seconds=perf_counter() - started_at,
                error="A non-empty prompt is required.",
            )
        if len(normalized_prompt) > MAX_PROMPT_CHARACTERS:
            return GenerationResult(
                text="",
                provider=self.provider_name,
                mode="SIMULATED",
                model=selected_model,
                simulated=True,
                elapsed_seconds=perf_counter() - started_at,
                error=(
                    f"Prompts are limited to {MAX_PROMPT_CHARACTERS:,} characters."
                ),
            )

        response_body = self._build_simulated_response(normalized_prompt)
        text = (
            f"{SIMULATION_DISCLOSURE}\n\n"
            f"{response_body}\n\n"
            "This portfolio-safe response was generated by the application's "
            "built-in Demo Provider and does not represent live model inference."
        )
        return GenerationResult(
            text=text,
            provider=self.provider_name,
            mode="SIMULATED",
            model=selected_model,
            simulated=True,
            elapsed_seconds=perf_counter() - started_at,
        )

    @staticmethod
    def _build_simulated_response(prompt: str) -> str:
        normalized = _normalize_demo_prompt(prompt)

        if any(term in normalized for term in ("local ai", "local inference")):
            return (
                "Local inference runs a model on hardware you control. It can "
                "improve privacy and offline availability, while requiring enough "
                "local memory, storage, and compute for the selected model."
            )

        topic = _match_demo_topic(normalized)
        if topic:
            return DEMO_KNOWLEDGE_BASE[topic]
        return UNKNOWN_DEMO_TOPIC_RESPONSE


def _normalize_demo_prompt(prompt: str) -> str:
    """Reduce harmless wording differences without interpreting the prompt."""

    return " ".join(re.findall(r"[a-z0-9]+", prompt.casefold()))


def _match_demo_topic(normalized_prompt: str) -> Optional[str]:
    """Match reviewed aliases, allowing only conservative single-word typos."""

    padded_prompt = f" {normalized_prompt} "
    prompt_words = normalized_prompt.split()

    for topic, aliases in DEMO_TOPIC_ALIASES.items():
        for alias in aliases:
            if f" {alias} " in padded_prompt:
                return topic

    typo_candidates = [
        (topic, alias)
        for topic, aliases in DEMO_TOPIC_ALIASES.items()
        for alias in aliases
        if " " not in alias and len(alias) >= 5
    ]
    for word in prompt_words:
        if len(word) < 4:
            continue
        for topic, alias in typo_candidates:
            if SequenceMatcher(None, word, alias).ratio() >= 0.83:
                return topic
    return None


class OllamaProvider:
    """Provider-contract adapter around the backwards-compatible Ollama client."""

    provider_name = "Local Ollama"

    def __init__(self, settings: Mapping[str, Any]) -> None:
        from src.ai_client import OllamaClient

        base_url = str(settings.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        _validate_ollama_url(base_url)
        self.client = OllamaClient(
            base_url=base_url,
            selected_model=str(settings.get("OLLAMA_MODEL", "llama3.2")),
            health_timeout_seconds=_positive_float_setting(
                settings, "OLLAMA_HEALTH_TIMEOUT", 2.5
            ),
            generation_timeout_seconds=_positive_float_setting(
                settings, "OLLAMA_GENERATION_TIMEOUT", 120.0
            ),
        )

    @property
    def selected_model(self) -> str:
        return self.client.selected_model

    @selected_model.setter
    def selected_model(self, value: str) -> None:
        self.client.selected_model = value

    def list_models(self) -> List[str]:
        return self.client.list_models()

    def get_readiness(self) -> ProviderReadiness:
        return self.client.get_provider_readiness()

    def generate(self, prompt: str, model_name: Optional[str] = None) -> GenerationResult:
        return self.client.generate_result(prompt, model_name)


ProviderBuilder = Callable[[Mapping[str, Any]], InferenceProvider]


class ProviderFactory:
    """Create providers without coupling the UI to a concrete implementation."""

    def __init__(self) -> None:
        self._builders: Dict[str, ProviderBuilder] = {
            DEFAULT_PROVIDER: lambda _settings: DemoProvider(),
            "ollama": _build_ollama_provider,
        }

    def register(self, name: str, builder: ProviderBuilder) -> None:
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ProviderConfigurationError("Provider names cannot be empty.")
        self._builders[normalized_name] = builder

    def create(
        self,
        provider_name: Optional[str] = None,
        settings: Optional[Mapping[str, Any]] = None,
        environment: Optional[Mapping[str, str]] = None,
    ) -> InferenceProvider:
        resolved_name = resolve_provider_name(provider_name, settings, environment)
        builder = self._builders.get(resolved_name)
        if builder is None:
            available = ", ".join(sorted(self._builders))
            raise ProviderConfigurationError(
                f"Provider '{resolved_name}' is not configured. Available providers: {available}."
            )
        return builder(settings or {})


def resolve_provider_name(
    provider_name: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve explicit settings first and default safely to Demo Mode."""

    if provider_name and provider_name.strip():
        return provider_name.strip().lower()

    configured_name = (settings or {}).get(PROVIDER_ENV_VAR)
    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip().lower()

    env_name = (environment if environment is not None else os.environ).get(PROVIDER_ENV_VAR)
    if env_name and env_name.strip():
        return env_name.strip().lower()

    return DEFAULT_PROVIDER


def create_provider(
    provider_name: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> InferenceProvider:
    """Convenience entry point used by the application composition root."""

    return ProviderFactory().create(provider_name, settings, environment)


def _build_ollama_provider(settings: Mapping[str, Any]) -> InferenceProvider:
    """Build local Ollama lazily so Demo Mode has no local-service dependency."""
    return OllamaProvider(settings)


def _positive_float_setting(
    settings: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    raw_value = settings.get(name, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(f"{name} must be a number.") from exc
    if not isfinite(value) or value <= 0:
        raise ProviderConfigurationError(f"{name} must be greater than zero.")
    return value


def _validate_ollama_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderConfigurationError(
            "OLLAMA_BASE_URL must be an absolute HTTP or HTTPS URL."
        )
    if parsed.username or parsed.password:
        raise ProviderConfigurationError(
            "OLLAMA_BASE_URL must not contain embedded credentials."
        )
