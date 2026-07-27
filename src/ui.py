from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src.conversation import get_example_prompts
from src.providers import ProviderReadiness


def apply_theme() -> None:
    """Apply a polished blue enterprise appearance without changing the app identity."""
    st.markdown(
        """
        <style>
            :root {
                --primary: #2563eb;
                --surface: #ffffff;
                --text: #0f172a;
            }
            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #f1f6ff 100%);
                color: var(--text);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .status-card, .metric-card, .chat-card {
                background: var(--surface);
                border: 1px solid #dbeafe;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
                padding: 1.1rem 1.15rem;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the main title and subtitle."""
    st.title("Localized AI Inference Sandbox")
    st.caption(
        "A safe portfolio demonstration of local AI inference, system monitoring, and graceful fallback behavior."
    )


def render_status_card(readiness: ProviderReadiness) -> None:
    """Render the readiness-based status card."""
    with st.container():
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.subheader("Environment status")
        st.write(f"Provider: {readiness.provider}")
        st.write(f"Provider status: {status_label(readiness.ready)}")
        st.write(f"Model status: {model_status_label(readiness)}")
        st.write(f"Current mode: {readiness.mode}")
        if readiness.simulated:
            st.info("Demo output is simulated and is not produced by a real AI model.")
        st.write(readiness.message)
        st.markdown("</div>", unsafe_allow_html=True)


def render_model_selector(readiness: ProviderReadiness) -> None:
    """Render a model selection dropdown and refresh button."""
    with st.container():
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.subheader("Model management")

        models = readiness.available_models
        is_disabled = not models

        index = 0
        current_model = st.session_state.get("selected_model")
        if current_model in models:
            index = models.index(current_model)

        selected = st.selectbox(
            "Select model",
            options=models if models else ["No models found"],
            index=index if models else 0,
            disabled=is_disabled,
            key="model_selector_widget",
        )

        if not is_disabled:
            st.session_state.selected_model = selected

        if st.button("Refresh models"):
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def status_label(reachable: bool) -> str:
    return "Connected" if reachable else "Not connected"


def model_status_label(readiness: ProviderReadiness) -> str:
    if not readiness.ready and not readiness.available_models:
        return "Unavailable"
    if not readiness.available_models:
        return "No models installed"
    if readiness.selected_model in readiness.available_models:
        return "Ready"
    return "Selected model missing"


def render_metric_cards(snapshot: Dict[str, Any]) -> None:
    """Render four metric cards for the application host runtime."""
    st.caption("Metrics describe the server running this app, not the visitor's device.")
    metrics = [
        ("CPU usage", f"{snapshot['cpu_usage']:.1f}%"),
        ("RAM usage", f"{snapshot['ram_usage']:.1f}%"),
        ("Disk usage", f"{snapshot['disk_usage']:.1f}%"),
        ("Available RAM", f"{snapshot['available_ram']:.2f} GB"),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=label, value=value)
            st.markdown("</div>", unsafe_allow_html=True)


def render_system_details(snapshot: Dict[str, Any]) -> None:
    """Render additional application-host details in a compact card."""
    with st.container():
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.subheader("Application runtime details")
        st.write(f"Operating system: {snapshot['operating_system']}")
        st.write(f"Python version: {snapshot['python_version']}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_prompt_examples() -> None:
    """Render clickable example prompt buttons that populate the prompt flow."""
    prompts = get_example_prompts()
    st.write("Example prompts")
    for prompt in prompts:
        if st.button(prompt, key=f"prompt_{prompt}"):
            st.session_state.prompt_input = prompt
