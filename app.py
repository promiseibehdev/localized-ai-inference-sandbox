from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st

from src.conversation import add_conversation_message, clear_conversation, initialize_conversation_state
from src.providers import (
    InferenceProvider,
    MAX_PROMPT_CHARACTERS,
    ProviderConfigurationError,
    ProviderReadiness,
    create_provider,
)
from src.system_monitor import get_system_snapshot
from src.ui import apply_theme, render_header, render_metric_cards, render_model_selector, render_prompt_examples, render_status_card, render_system_details


PROVIDER_SETTING_KEYS = {
    "INFERENCE_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_HEALTH_TIMEOUT",
    "OLLAMA_GENERATION_TIMEOUT",
}


def load_provider_settings() -> Dict[str, Any]:
    """Load deployment secrets without requiring a secrets file to exist."""
    settings: Dict[str, Any] = {}
    try:
        for key in PROVIDER_SETTING_KEYS:
            if key in st.secrets:
                settings[key] = st.secrets[key]
    except (FileNotFoundError, OSError):
        pass

    for key in PROVIDER_SETTING_KEYS:
        if key not in settings and key in os.environ:
            settings[key] = os.environ[key]
    return settings


def build_provider() -> InferenceProvider:
    """Create the configured provider; no configuration safely means Demo Mode."""
    try:
        return create_provider(settings=load_provider_settings())
    except (ProviderConfigurationError, TypeError, ValueError):
        st.warning(
            "The configured inference provider is invalid, so the application "
            "started in safe Demo Mode. Review the private provider settings."
        )
        return create_provider("demo", environment={})


def synchronize_selected_model(provider: InferenceProvider) -> None:
    """Apply widget/session selection before the single readiness request."""
    widget_selection = st.session_state.get("model_selector_widget")
    if widget_selection and widget_selection != "No models found":
        st.session_state.selected_model = widget_selection

    selected_model = st.session_state.get("selected_model")
    if selected_model:
        provider.selected_model = selected_model


def initialize_first_available_model(
    provider: InferenceProvider,
    readiness: ProviderReadiness,
) -> bool:
    """Select the first model once and request a clean rerun when needed."""
    current_model = st.session_state.get("selected_model")
    available_models = readiness.available_models
    if available_models and current_model not in available_models:
        selected_model = (
            readiness.selected_model
            if readiness.selected_model in available_models
            else available_models[0]
        )
        st.session_state.selected_model = selected_model
        provider.selected_model = selected_model
        return selected_model != readiness.selected_model
    return False


def main() -> None:
    st.set_page_config(page_title="Localized AI Inference Sandbox", page_icon="🤖", layout="wide")
    apply_theme()
    initialize_conversation_state()
    render_header()

    provider = build_provider()
    synchronize_selected_model(provider)
    readiness = provider.get_readiness()
    if initialize_first_available_model(provider, readiness):
        st.rerun()

    system_snapshot = get_system_snapshot()

    left, right = st.columns([1.2, 0.8])

    with left:
        render_status_card(readiness)
        render_model_selector(readiness)
        render_metric_cards(system_snapshot)
        render_system_details(system_snapshot)

    with right:
        st.markdown('<div class="chat-card">', unsafe_allow_html=True)
        st.subheader("Conversation")
        st.write("Use the prompt box below to try a live or demo response.")
        for item in st.session_state.conversation_history:
            if item["role"] == "user":
                st.chat_message("user").write(item["content"])
            else:
                message = item["content"]
                meta = []
                if item.get("mode"):
                    meta.append(f"Mode: {item['mode']}")
                if item.get("provider"):
                    meta.append(f"Provider: {item['provider']}")
                if "response_time" in item:
                    meta.append(f"Response time: {item['response_time']:.2f}s")
                with st.chat_message("assistant"):
                    if item.get("simulated"):
                        st.warning("Simulated output - no real AI model was used.")
                    st.write(message)
                    st.caption(" | ".join(meta))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.subheader("Prompt flow")
        render_prompt_examples()
        with st.form("prompt_form", clear_on_submit=True):
            prompt = st.text_input(
                "Enter a prompt",
                key="prompt_input",
                max_chars=MAX_PROMPT_CHARACTERS,
            )
            submitted = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            normalized_prompt = prompt.strip()
            if normalized_prompt:
                add_conversation_message("user", normalized_prompt)
                result = provider.generate(
                    normalized_prompt,
                    st.session_state.get("selected_model") or None,
                )
                add_conversation_message(
                    "assistant",
                    result.text,
                    mode=result.mode,
                    response_time=result.elapsed_seconds,
                    provider=result.provider,
                    simulated=result.simulated,
                    error=result.error or "",
                )
                st.rerun()
            else:
                st.warning("Enter a prompt before sending.")

        if st.button("Clear conversation"):
            clear_conversation()
            st.rerun()


if __name__ == "__main__":
    main()
