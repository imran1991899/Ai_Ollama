import json
import requests
import streamlit as st

# Public guest endpoint (no login, no API key required)
GUEST_API_URL = "https://text.pollinations.ai/openai"

# Available free guest models
GUEST_MODELS = [
    "openai",       # Default fast general model
    "openai-large", # Larger reasoning/coding model
    "qwen-coder",   # Code-focused model
    "llama",        # Meta Llama model
    "mistral",      # Mistral model
]

st.set_page_config(
    page_title="RepoAI // Guest Cloud Studio",
    page_icon="🐙",
    layout="wide",
)

# GitHub Dark-Mode Styling
st.markdown(
    """
    <style>
        .gh-header {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 16px;
        }
        .gh-repo-path {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 1.1rem;
            color: #58a6ff;
            font-weight: 600;
        }
        .gh-badge {
            display: inline-block;
            border: 1px solid #30363d;
            border-radius: 2em;
            padding: 2px 10px;
            font-size: 0.75rem;
            color: #8b949e;
            margin-left: 8px;
        }
        .gh-badge-green {
            display: inline-block;
            border: 1px solid #2ea04366;
            background-color: #2ea04326;
            border-radius: 2em;
            padding: 2px 10px;
            font-size: 0.75rem;
            color: #3fb950;
            margin-left: 8px;
        }
        div[data-testid="stChatMessage"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def stream_guest_chat(model: str, messages: list[dict], temperature: float, system_prompt: str):
    payload_messages = []
    if system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})
    payload_messages.extend(messages)

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "stream": True,
    }

    with requests.post(
        GUEST_API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=90,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
            except json.JSONDecodeError:
                continue


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Settings
with st.sidebar:
    st.markdown("### ⚙️ Repository Settings")
    selected_model = st.selectbox("Guest Model", GUEST_MODELS, index=0)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    system_prompt = st.text_area(
        "System Instructions (.ai-rules)",
        value="You are a concise, helpful engineering assistant. Format code clearly with language tags.",
        height=120,
    )

    st.divider()
    if st.button("🗑️ Clear Thread", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Mode: Keyless Guest Access (No Login)")

# GitHub-Style Repo Header
st.markdown(
    f"""
    <div class="gh-header">
        <span class="gh-repo-path">guest-dev / personal-ai</span>
        <span class="gh-badge-green">● Online (No Key)</span>
        <span class="gh-badge">Model: {selected_model}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display Chat History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat Input & Response Streaming
if prompt := st.chat_input("Ask anything straight away (no login required)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            full_response = st.write_stream(
                stream_guest_chat(
                    model=selected_model,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            )
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.warning("Empty response returned. Try sending the message again or switching models.")
        except requests.RequestException as exc:
            st.error(f"Guest endpoint busy or unreachable. Try again in a few seconds.\n\nDetails: `{exc}`")
