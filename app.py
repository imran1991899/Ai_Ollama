import base64
import json
import fitz  # PyMuPDF for PDF text extraction
import requests
import streamlit as st

# Public guest endpoint (no login, no API key required)
GUEST_API_URL = "https://text.pollinations.ai/openai"

# Vision-capable models first so pasted/uploaded images work out of the box
GUEST_MODELS = [
    "openai",       # Supports text + pasted/uploaded images
    "openai-large", # Larger reasoning + vision model
    "qwen-coder",   # Code-focused model (best for text/code files)
    "llama",        # Meta Llama model
    "mistral",      # Mistral model
]

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

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


def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from PDF or plain-text/code files."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = [page.get_text("text") for page in doc]
            doc.close()
            extracted = "\n".join(pages).strip()
            return extracted if extracted else "[PDF contained no selectable text]"
        except Exception as exc:
            return f"[Could not parse PDF: {exc}]"

    # Treat remaining non-image files as UTF-8 text/code
    return raw_bytes.decode("utf-8", errors="replace")


def parse_uploaded_items(files: list) -> tuple[list[dict], list[str], list[bytes]]:
    """
    Processes uploaded/pasted files into:
    - api_content_parts: OpenAI-compatible multimodal content blocks
    - attachment_labels: File names for display in chat history
    - preview_images: Raw image bytes for rendering thumbnails in chat
    """
    api_content_parts = []
    attachment_labels = []
    preview_images = []

    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        raw_bytes = f.getvalue()

        if ext in IMAGE_EXTENSIONS or (f.type and f.type.startswith("image/")):
            mime = f.type or f"image/{'jpeg' if ext == 'jpg' else ext or 'png'}"
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            api_content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            attachment_labels.append(f"🖼️ {f.name}")
            preview_images.append(raw_bytes)
        else:
            file_text = extract_text_from_file(f)
            # Cap very large text files so guest payload stays within limits
            if len(file_text) > 25000:
                file_text = file_text[:25000] + "\n\n...[truncated due to length]..."
            api_content_parts.append(
                {
                    "type": "text",
                    "text": f"--- Attached File: {f.name} ---\n{file_text}\n--- End of {f.name} ---",
                }
            )
            attachment_labels.append(f"📄 {f.name}")

    return api_content_parts, attachment_labels, preview_images


def stream_guest_chat(model: str, messages: list[dict], temperature: float, system_prompt: str):
    payload_messages = []
    if system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})

    for msg in messages:
        payload_messages.append({"role": msg["role"], "content": msg["api_content"]})

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

    st.caption("Tip: Click the chat box and press Ctrl+V to paste screenshots, or click + to upload files.")

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
        if msg.get("attachments"):
            st.caption("Attachments: " + " • ".join(msg["attachments"]))
        for img_bytes in msg.get("images", []):
            st.image(img_bytes, width=320)
        if msg.get("display_text"):
            st.markdown(msg["display_text"])

# Chat Input with File Upload + Clipboard Image Paste (Ctrl+V)
submission = st.chat_input(
    "Ask anything, paste an image (Ctrl+V), or upload files...",
    accept_file="multiple",
)

if submission:
    user_text = submission.text.strip()
    uploaded_files = submission.files or []

    if not user_text and not uploaded_files:
        st.stop()

    api_parts, attachment_labels, preview_images = parse_uploaded_items(uploaded_files)

    # Build OpenAI-compatible content (string if text-only, list of parts if files/images attached)
    if api_parts:
        prompt_text = user_text if user_text else "Analyze the attached file(s) / image(s)."
        api_content = [{"type": "text", "text": prompt_text}] + api_parts
    else:
        api_content = user_text

    user_msg = {
        "role": "user",
        "display_text": user_text,
        "api_content": api_content,
        "attachments": attachment_labels,
        "images": preview_images,
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="🧑‍💻"):
        if attachment_labels:
            st.caption("Attachments: " + " • ".join(attachment_labels))
        for img_bytes in preview_images:
            st.image(img_bytes, width=320)
        if user_text:
            st.markdown(user_text)

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
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "display_text": full_response,
                        "api_content": full_response,
                    }
                )
            else:
                st.warning("Empty response returned. Try sending again or switching models.")
        except requests.RequestException as exc:
            st.error(
                f"Guest endpoint busy or model does not support images. "
                f"Use `openai` or `openai-large` for images.\n\nDetails: `{exc}`"
            )
