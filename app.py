import streamlit as st

st.set_page_config(
    page_title="Localized AI Inference Sandbox",
    page_icon="🤖",
    layout="centered",
    menu_items={},
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] {display: none;}
        div[data-testid="stToolbar"] {display: none;}
        div[data-testid="stStatusWidget"] {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        footer {display: none;}
        .stDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Localized AI Inference Sandbox")
st.write("Project Foundation Successfully Created")
