from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.providers import GenerationResult, ProviderReadiness


class OllamaClient:
    """Thin client for interacting with a local Ollama service.

    The client supports live inference when Ollama is reachable and the selected
    model is installed. When inference is unavailable or fails, it falls back to
    a safe demo response so the app remains usable in offline or restricted
    environments.
    """

    provider_name = "Local Ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        selected_model: str = "llama3.2",
        health_timeout_seconds: float = 2.5,
        generation_timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.selected_model = selected_model
        self.health_timeout_seconds = health_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds
        # Kept as a compatibility alias for code that previously inspected it.
        self.timeout_seconds = health_timeout_seconds
        self._last_generation_error: Optional[str] = None

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = (
                self.generation_timeout_seconds
                if method.upper() == "POST" and path.rstrip("/") == "/api/generate"
                else self.health_timeout_seconds
            )
        if method.upper() == "GET":
            return requests.get(url, timeout=timeout, **kwargs)
        if method.upper() == "POST":
            return requests.post(url, timeout=timeout, **kwargs)
        return requests.request(method=method, url=url, timeout=timeout, **kwargs)

    def _fetch_models(self) -> Tuple[bool, List[str], Optional[str]]:
        """Fetch server and model readiness with one inexpensive request."""
        try:
            response = self._request("GET", "/api/tags")
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout:
            return False, [], "Ollama health check timed out."
        except requests.ConnectionError:
            return False, [], "Ollama is not reachable at the configured URL."
        except requests.HTTPError as exc:
            return False, [], f"Ollama health check returned an HTTP error: {exc}."
        except requests.RequestException:
            return False, [], "Ollama is not reachable."
        except (ValueError, TypeError):
            return False, [], "Ollama returned an invalid health response."

        if not isinstance(payload, dict):
            return False, [], "Ollama returned an invalid model-list response."

        raw_models = payload.get("models", [])
        if not isinstance(raw_models, list):
            return False, [], "Ollama returned an invalid model-list response."

        models: List[str] = []
        for item in raw_models:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name:
                    models.append(name)
        return True, models, None

    def is_reachable(self) -> bool:
        """Return True when the Ollama server responds quickly enough."""
        reachable, _, _ = self._fetch_models()
        return reachable

    def list_models(self) -> List[str]:
        """Return a normalized list of installed Ollama model names."""
        _, models, _ = self._fetch_models()
        return models

    def model_exists(self, model_name: Optional[str] = None) -> bool:
        """Return True when the requested model is present in the installed list."""
        target_name = model_name or self.selected_model
        return target_name in self.list_models()

    def generate(self, prompt: str, model_name: Optional[str] = None) -> str:
        """Generate a response from Ollama or fall back to a safe demo response."""
        return self.generate_result(prompt, model_name).text

    def generate_result(self, prompt: str, model_name: Optional[str] = None) -> GenerationResult:
        """Generate with structured timing and accurate failure information."""
        started_at = perf_counter()
        target_model = model_name or self.selected_model
        reachable, available_models, readiness_error = self._fetch_models()

        if not reachable:
            error = readiness_error or "Ollama is not reachable."
            return self._fallback_result(prompt, target_model, started_at, error)
        if target_model not in available_models:
            error = f"The selected Ollama model '{target_model}' is not installed."
            return self._fallback_result(prompt, target_model, started_at, error)

        try:
            response = self._request(
                "POST",
                "/api/generate",
                json={"model": target_model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout:
            return self._fallback_result(
                prompt, target_model, started_at, "Ollama generation timed out."
            )
        except requests.ConnectionError:
            return self._fallback_result(
                prompt, target_model, started_at, "Ollama disconnected during generation."
            )
        except requests.HTTPError as exc:
            return self._fallback_result(
                prompt, target_model, started_at, f"Ollama generation returned an HTTP error: {exc}."
            )
        except (requests.RequestException, ValueError, TypeError):
            return self._fallback_result(
                prompt, target_model, started_at, "Ollama returned an invalid generation response."
            )

        text = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            return self._fallback_result(
                prompt, target_model, started_at, "Ollama returned an empty generation response."
            )

        self._last_generation_error = None
        return GenerationResult(
            text=text,
            provider=self.provider_name,
            mode="LIVE",
            model=target_model,
            simulated=False,
            elapsed_seconds=perf_counter() - started_at,
        )

    def _fallback_result(
        self,
        prompt: str,
        target_model: str,
        started_at: float,
        error: str,
    ) -> GenerationResult:
        self._last_generation_error = error
        return GenerationResult(
            text=self._demo_response(prompt, error),
            provider=self.provider_name,
            mode="DEMO",
            model=target_model,
            simulated=True,
            elapsed_seconds=perf_counter() - started_at,
            error=error,
        )

    def _demo_response(self, prompt: str, reason: Optional[str] = None) -> str:
        """Return a safe non-crashing demo response for unavailable inference."""
        detail = reason or "Ollama inference is unavailable."
        return (
            "SIMULATED DEMO RESPONSE - No real AI model was used. "
            f"Reason: {detail} Prompt received: {prompt}"
        )

    def get_readiness(self) -> Dict[str, Any]:
        """Return readiness using model discovery only, never inference."""
        ollama_reachable, available_models, readiness_error = self._fetch_models()
        selected_model = self.selected_model
        model_available = selected_model in available_models

        if not ollama_reachable:
            mode = "DEMO"
            message = f"Demo mode active: {readiness_error or 'Ollama is not reachable.'}"
            live_ready = False
        elif not available_models:
            mode = "DEMO"
            message = "Demo mode active: Ollama is connected, but no models are installed."
            live_ready = False
        elif not model_available:
            mode = "DEMO"
            message = "Demo mode active: The selected model is not installed."
            live_ready = False
        elif self._last_generation_error:
            mode = "DEMO"
            message = f"Demo mode active: {self._last_generation_error}"
            live_ready = False
        else:
            live_ready = True
            mode = "LIVE"
            message = "Live mode ready: Ollama is connected and the selected model is installed."

        return {
            "ollama_reachable": ollama_reachable,
            "available_models": available_models,
            "selected_model": selected_model,
            "model_available": model_available,
            "live_ready": live_ready,
            "mode": mode,
            "message": message,
        }

    def get_provider_readiness(self) -> ProviderReadiness:
        """Return the provider-neutral readiness contract used by the new UI."""
        readiness = self.get_readiness()
        return ProviderReadiness(
            provider=self.provider_name,
            mode=readiness["mode"],
            ready=readiness["live_ready"],
            simulated=readiness["mode"] != "LIVE",
            message=readiness["message"],
            available_models=readiness["available_models"],
            selected_model=readiness["selected_model"],
        )
