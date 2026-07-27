from unittest.mock import patch

import requests

from src.ai_client import OllamaClient


class DummyResponse:
    def __init__(self, payload=None, status_code=200, ok=True):
        self._payload = payload or {}
        self.status_code = status_code
        self.ok = ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError("request failed")


def test_ollama_client_imports_correctly():
    client = OllamaClient()
    assert isinstance(client, OllamaClient)


def test_ollama_client_has_get_readiness():
    client = OllamaClient()
    assert hasattr(client, "get_readiness")


def test_get_readiness_returns_dictionary():
    client = OllamaClient()
    readiness = client.get_readiness()
    assert isinstance(readiness, dict)


def test_get_readiness_contains_required_keys():
    client = OllamaClient()
    readiness = client.get_readiness()
    required_keys = {
        "ollama_reachable",
        "available_models",
        "selected_model",
        "model_available",
        "live_ready",
        "mode",
        "message",
    }
    assert required_keys == set(readiness.keys())


def test_demo_mode_works_when_ollama_is_unavailable():
    client = OllamaClient(selected_model="llama3.2")

    with patch("src.ai_client.requests.get", side_effect=requests.RequestException("offline")):
        with patch("src.ai_client.requests.post", side_effect=requests.RequestException("offline")):
            readiness = client.get_readiness()
            response = client.generate("hello")

    assert readiness["mode"] == "DEMO"
    assert readiness["live_ready"] is False
    assert "demo" in readiness["message"].lower()
    assert "demo" in response.lower()


def test_model_listing_works_with_mocked_responses():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}, {"name": "phi3"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        models = client.list_models()
        exists = client.model_exists("llama3.2")

    assert models == ["llama3.2", "phi3"]
    assert exists is True


def test_live_mode_is_only_returned_when_conditions_are_valid():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        with patch("src.ai_client.requests.post", return_value=DummyResponse({"response": "Live response"})):
            readiness = client.get_readiness()

    assert readiness["mode"] == "LIVE"
    assert readiness["live_ready"] is True
    assert readiness["model_available"] is True


def test_generation_failure_falls_back_safely_to_demo_mode():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        with patch("src.ai_client.requests.post", return_value=DummyResponse({}, status_code=500, ok=False)):
            response = client.generate("hello")
            readiness = client.get_readiness()

    assert "demo" in response.lower()
    assert readiness["mode"] == "DEMO"
    assert readiness["live_ready"] is False


def test_readiness_message_for_unreachable_ollama():
    client = OllamaClient(selected_model="llama3.2")

    with patch("src.ai_client.requests.get", side_effect=requests.RequestException("offline")):
        readiness = client.get_readiness()

    assert readiness["message"] == "Demo mode active: Ollama is not reachable."


def test_readiness_message_for_no_installed_models():
    client = OllamaClient(selected_model="llama3.2")

    with patch("src.ai_client.requests.get", return_value=DummyResponse({"models": []})):
        readiness = client.get_readiness()

    assert readiness["message"] == "Demo mode active: Ollama is connected, but no models are installed."


def test_readiness_message_for_missing_selected_model():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "phi3"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        readiness = client.get_readiness()

    assert readiness["message"] == "Demo mode active: The selected model is not installed."


def test_readiness_message_for_live_mode():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        with patch("src.ai_client.requests.post", return_value=DummyResponse({"response": "Live response"})):
            readiness = client.get_readiness()

    assert readiness["message"] == "Live mode ready: Ollama is connected and the selected model is installed."


def test_readiness_never_runs_inference():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        with patch("src.ai_client.requests.post") as mocked_post:
            readiness = client.get_readiness()

    assert readiness["mode"] == "LIVE"
    mocked_post.assert_not_called()


def test_health_and_generation_use_separate_timeouts():
    client = OllamaClient(
        selected_model="llama3.2",
        health_timeout_seconds=1.25,
        generation_timeout_seconds=45.0,
    )
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)) as mocked_get:
        with patch(
            "src.ai_client.requests.post",
            return_value=DummyResponse({"response": "Live response"}),
        ) as mocked_post:
            result = client.generate_result("hello")

    assert result.mode == "LIVE"
    assert mocked_get.call_args.kwargs["timeout"] == 1.25
    assert mocked_post.call_args.kwargs["timeout"] == 45.0


def test_generation_timeout_returns_accurate_simulated_result():
    client = OllamaClient(selected_model="llama3.2")
    payload = {"models": [{"name": "llama3.2"}]}

    with patch("src.ai_client.requests.get", return_value=DummyResponse(payload)):
        with patch("src.ai_client.requests.post", side_effect=requests.Timeout("slow")):
            result = client.generate_result("hello")

    assert result.mode == "DEMO"
    assert result.simulated is True
    assert result.error == "Ollama generation timed out."
    assert "No real AI model was used" in result.text
