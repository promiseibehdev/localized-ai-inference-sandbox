from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class OllamaClient:
    """Thin client for interacting with a local Ollama service.

    The client supports live inference when Ollama is reachable and the selected
    model is installed. When inference is unavailable or fails, it falls back to
    a safe demo response so the app remains usable in offline or restricted
    environments.
    """

    def __init__(self, base_url: str = "http://localhost:11434", selected_model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.selected_model = selected_model
        self.timeout_seconds = 2.5

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if method.upper() == "GET":
            return requests.get(url, timeout=self.timeout_seconds, **kwargs)
        if method.upper() == "POST":
            return requests.post(url, timeout=self.timeout_seconds, **kwargs)
        return requests.request(method=method, url=url, timeout=self.timeout_seconds, **kwargs)

    def is_reachable(self) -> bool:
        """Return True when the Ollama server responds quickly enough."""
        try:
            response = self._request("GET", "/api/tags")
            response.raise_for_status()
            return True
        except (requests.RequestException, ValueError):
            return False

    def list_models(self) -> List[str]:
        """Return a normalized list of installed Ollama model names."""
        try:
            response = self._request("GET", "/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            return []

        models: List[str] = []
        for item in payload.get("models", []):
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name:
                    models.append(name)
        return models

    def model_exists(self, model_name: Optional[str] = None) -> bool:
        """Return True when the requested model is present in the installed list."""
        target_name = model_name or self.selected_model
        return target_name in self.list_models()

    def generate(self, prompt: str, model_name: Optional[str] = None) -> str:
        """Generate a response from Ollama or fall back to a safe demo response."""
        target_model = model_name or self.selected_model

        if not self.is_reachable() or not self.model_exists(target_model):
            return self._demo_response(prompt)

        try:
            response = self._request(
                "POST",
                "/api/generate",
                json={"model": target_model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            return self._demo_response(prompt)

        text = payload.get("response")
        if isinstance(text, str) and text.strip():
            return text
        return self._demo_response(prompt)

    def _demo_response(self, prompt: str) -> str:
        """Return a safe non-crashing demo response for unavailable inference."""
        return f"Demo response: Ollama inference is unavailable. Prompt received: {prompt}"

    def get_readiness(self) -> Dict[str, Any]:
        """Return a readiness dictionary describing the current Ollama state."""
        ollama_reachable = self.is_reachable()
        available_models = self.list_models()
        selected_model = self.selected_model
        model_available = selected_model in available_models

        if not ollama_reachable:
            mode = "DEMO"
            message = "Demo mode active: Ollama is not reachable."
            live_ready = False
        elif not available_models:
            mode = "DEMO"
            message = "Demo mode active: Ollama is connected, but no models are installed."
            live_ready = False
        elif not model_available:
            mode = "DEMO"
            message = "Demo mode active: The selected model is not installed."
            live_ready = False
        else:
            try:
                response = self._request(
                    "POST",
                    "/api/generate",
                    json={"model": selected_model, "prompt": "ready", "stream": False},
                )
                response.raise_for_status()
                payload = response.json()
                text = payload.get("response")
                live_ready = isinstance(text, str) and bool(text.strip())
            except (requests.RequestException, ValueError, TypeError):
                live_ready = False

            if live_ready:
                mode = "LIVE"
                message = "Live mode active."
            else:
                mode = "DEMO"
                message = "Demo mode active: The selected model is not installed."

        return {
            "ollama_reachable": ollama_reachable,
            "available_models": available_models,
            "selected_model": selected_model,
            "model_available": model_available,
            "live_ready": live_ready,
            "mode": mode,
            "message": message,
        }
