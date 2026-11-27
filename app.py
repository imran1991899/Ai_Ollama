import json
import requests
import streamlit as st

OLLAMA_BASE_URL = "http://localhost:11434"

st.set_page_config(
    page_title="RepoAI // Local Ollama Studio",
    page_icon="🐙",
    layout="wide",
)

# Custom GitHub-inspired borders, repo header bar, and code blocks
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


@st.cache_data(ttl=10)
def get_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return models if models else ["llama3.2"]
    except requests.RequestException:
        return ["llama3.2"]


def stream_ollama_chat(model: str, messages: list[dict], temperature: float, system_prompt: str):
    payload_messages = []
    if system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})
    payload_messages.extend(messages)

    payload = {
        "model": model,
        "messages": payload_messages,
        "stream": True,
        "options": {"temperature": temperature},
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content


# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar ("Issues / Settings" panel)
with st.sidebar:
    st.markdown("### ⚙️ Repository Settings")
    available_models = get_ollama_models()
    selected_model = st.selectbox("Ollama Model", available_models, index=0)

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

    st.caption(f"Endpoint: `{OLLAMA_BASE_URL}`")

# Top GitHub-style header bar
st.markdown(
    f"""
    <div class="gh-header">
        <span class="gh-repo-path">local-dev / personal-ai</span>
        <span class="gh-badge">Public</span>
        <span class="gh-badge">Model: {selected_model}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render chat history
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your local AI, paste code, or draft a commit/doc..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            full_response = st.write_stream(
                stream_ollama_chat(
                    model=selected_model,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except requests.RequestException as exc:
            err_msg = (
                f"Could not reach Ollama at `{OLLAMA_BASE_URL}`. "
                f"Make sure Ollama is running (`ollama serve`) and model `{selected_model}` is pulled (`ollama pull {selected_model}`).\n\n"
                f"Details: `{exc}`"
            )
            st.error(err_msg)
