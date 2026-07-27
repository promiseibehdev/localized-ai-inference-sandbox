from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


MAX_CONVERSATION_MESSAGES = 50


def initialize_conversation_state() -> None:
    """Initialize Streamlit session state for conversation state."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = ""


def add_conversation_message(
    role: str,
    content: str,
    mode: str = "",
    response_time: float = 0.0,
    provider: str = "",
    simulated: bool = False,
    error: str = "",
) -> None:
    """Append a chat message entry to the conversation history."""
    initialize_conversation_state()
    entry: Dict[str, Any] = {"role": role, "content": content}
    if mode:
        entry["mode"] = mode
    if response_time >= 0 and role == "assistant":
        entry["response_time"] = response_time
    if provider:
        entry["provider"] = provider
    if simulated:
        entry["simulated"] = True
    if error:
        entry["error"] = error
    st.session_state.conversation_history.append(entry)
    st.session_state.conversation_history = st.session_state.conversation_history[
        -MAX_CONVERSATION_MESSAGES:
    ]


def clear_conversation() -> None:
    """Clear all chat messages while preserving the other state keys."""
    initialize_conversation_state()
    st.session_state.conversation_history = []


def get_example_prompts() -> List[str]:
    """Return the example prompts used by the conversation interface."""
    return [
        "Explain local AI inference in simple terms.",
        "Summarize the benefits of running models locally.",
        "Describe how this app handles fallback behavior.",
    ]
