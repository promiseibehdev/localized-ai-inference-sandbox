from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import app
from src.ai_client import OllamaClient
from src.conversation import (
    MAX_CONVERSATION_MESSAGES,
    add_conversation_message,
    clear_conversation,
    initialize_conversation_state,
)
from src.system_monitor import get_system_snapshot
from src import ui


def test_system_monitor_returns_expected_keys():
    snapshot = get_system_snapshot()
    expected_keys = {
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "available_ram",
        "operating_system",
        "python_version",
    }
    assert expected_keys.issubset(snapshot.keys())


def test_system_monitor_handles_failures_safely():
    with patch("src.system_monitor.psutil.cpu_percent", side_effect=RuntimeError("boom")):
        with patch("src.system_monitor.psutil.virtual_memory", side_effect=RuntimeError("boom")):
            with patch("src.system_monitor.psutil.disk_usage", side_effect=RuntimeError("boom")):
                snapshot = get_system_snapshot()
    assert snapshot["cpu_usage"] >= 0
    assert snapshot["ram_usage"] >= 0
    assert snapshot["disk_usage"] >= 0
    assert snapshot["available_ram"] >= 0
    assert isinstance(snapshot["operating_system"], str)
    assert isinstance(snapshot["python_version"], str)


def test_conversation_state_initializes_correctly():
    st.session_state.clear()
    initialize_conversation_state()
    assert st.session_state["conversation_history"] == []
    assert st.session_state["selected_model"] == ""


def test_messages_can_be_added_and_cleared():
    st.session_state.clear()
    initialize_conversation_state()
    add_conversation_message("user", "hello")
    add_conversation_message("assistant", "hi", mode="DEMO", response_time=12.3)
    assert len(st.session_state["conversation_history"]) == 2
    clear_conversation()
    assert st.session_state["conversation_history"] == []


def test_conversation_history_is_bounded_for_public_sessions():
    st.session_state.clear()
    initialize_conversation_state()
    for index in range(MAX_CONVERSATION_MESSAGES + 5):
        add_conversation_message("user", f"message-{index}")

    history = st.session_state["conversation_history"]
    assert len(history) == MAX_CONVERSATION_MESSAGES
    assert history[0]["content"] == "message-5"


def test_ui_modules_import_correctly():
    assert ui is not None


def test_app_imports_without_importerror():
    assert app is not None


def test_ollama_client_still_has_get_readiness():
    client = OllamaClient()
    assert hasattr(client, "get_readiness")


def test_no_module_level_get_readiness_helper_exists():
    import src.ai_client as ai_client

    assert not hasattr(ai_client, "get_readiness_helper")


def test_only_one_ollama_client_class_exists_in_project():
    project_root = Path(__file__).resolve().parents[1]
    source_files = list((project_root / "src").glob("*.py"))
    count = 0
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        count += text.count("class OllamaClient")
    assert count == 1


def test_default_provider_is_demo_without_configuration():
    with patch.object(app, "load_provider_settings", return_value={}):
        provider = app.build_provider()

    assert provider.provider_name == "Built-in Demo"
    assert provider.get_readiness().mode == "SIMULATED"


def test_invalid_provider_configuration_falls_back_to_safe_demo():
    with patch.object(
        app,
        "load_provider_settings",
        return_value={"INFERENCE_PROVIDER": "unsupported"},
    ):
        provider = app.build_provider()

    assert provider.provider_name == "Built-in Demo"
    assert provider.get_readiness().simulated is True


def test_streamlit_app_starts_without_ollama_or_secrets():
    app_test = AppTest.from_file(str(Path(app.__file__)), default_timeout=10)
    app_test.run()

    assert not app_test.exception
    assert any("Built-in Demo" in item.value for item in app_test.markdown)


def test_prompt_submission_uses_demo_provider_without_widget_state_error():
    app_test = AppTest.from_file(str(Path(app.__file__)), default_timeout=10)
    app_test.run()
    app_test.text_input(key="prompt_input").input("Explain local AI inference.")
    app_test.button(key="FormSubmitter:prompt_form-Send").click().run()

    assert not app_test.exception
    assert any("SIMULATED DEMO RESPONSE" in item.value for item in app_test.markdown)


def test_response_time_uses_milliseconds_below_one_second():
    assert app.format_response_time(0.1234) == "123 ms"
    assert app.format_response_time(0.0004) == "0 ms"


def test_response_time_uses_seconds_at_or_above_one_second():
    assert app.format_response_time(1.234) == "1.23s"
