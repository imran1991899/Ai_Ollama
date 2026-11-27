import io
import json
import fitz  # PyMuPDF for PDF text extraction
import requests
import streamlit as st
from PIL import Image

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Public guest endpoint (no login, no API key required)
GUEST_API_URL = "https://text.pollinations.ai/openai"

GUEST_MODELS = [
    "openai",
    "openai-large",
    "qwen-coder",
    "llama",
    "mistral",
]

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}

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


def extract_text_from_image(raw_bytes: bytes) -> str:
    """Runs OCR on pasted/uploaded image bytes so text/timestamps/numbers are readable."""
    if not HAS_TESSERACT:
        return "[pytesseract not installed; image attached without OCR text]"
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        w, h = img.size
        if max(w, h) < 800:
            img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(img).strip()
        return text if text else "[No readable text detected in image]"
    except Exception as exc:
        return f"[OCR failed: {exc}]"


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

    return raw_bytes.decode("utf-8", errors="replace")


def build_prompt_with_attachments(user_text: str, files: list) -> tuple[str, list[str], list[bytes]]:
    """
    Converts uploaded/pasted files into a clean text prompt block for the guest API,
    returning (combined_prompt, attachment_labels, preview_images).
    """
    attachment_labels = []
    preview_images = []
    context_blocks = []

    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        raw_bytes = f.getvalue()

        if ext in IMAGE_EXTENSIONS or (f.type and f.type.startswith("image/")):
            ocr_text = extract_text_from_image(raw_bytes)
            attachment_labels.append(f"🖼️ {f.name}")
            preview_images.append(raw_bytes)
            context_blocks.append(
                f"[Pasted/Uploaded Image: {f.name}]\nOCR Extracted Text:\n{ocr_text}"
            )
        else:
            file_text = extract_text_from_file(f)
            if len(file_text) > 25000:
                file_text = file_text[:25000] + "\n\n...[truncated due to length]..."
            attachment_labels.append(f"📄 {f.name}")
            context_blocks.append(f"[Attached File: {f.name}]\n{file_text}")

    instruction = user_text.strip() if user_text.strip() else "Process the attached content."
    if context_blocks:
        combined = instruction + "\n\n" + "\n\n".join(context_blocks)
    else:
        combined = instruction

    return combined, attachment_labels, preview_images


def stream_guest_chat(model: str, messages: list[dict], temperature: float, system_prompt: str):
    payload_messages = []
    if system_prompt.strip():
        payload_messages.append({"role": "system", "content": system_prompt.strip()})

    for msg in messages:
        content_str = msg.get("api_content") or msg.get("display_text") or ""
        if isinstance(content_str, str) and content_str.strip():
            payload_messages.append({"role": msg["role"], "content": content_str})

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
        if resp.status_code == 400:
            err_detail = resp.text[:300]
            raise RuntimeError(f"400 Bad Request from guest endpoint: {err_detail}")
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
        value=0.3,
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

    st.caption("Paste images (Ctrl+V) or upload PDF/code/text files.")

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

    combined_prompt, attachment_labels, preview_images = build_prompt_with_attachments(
        user_text=user_text,
        files=uploaded_files,
    )

    user_msg = {
        "role": "user",
        "display_text": user_text or "Processed attached file/image.",
        "api_content": combined_prompt,
        "attachments": attachment_labels,
        "images": preview_images,
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="🧑‍💻"):
        if attachment_labels:
            st.caption("Attachments: " + " • ".join(attachment_labels))
        for img_bytes in preview_images:
            st.image(img_bytes, width=320)
        if user_msg["display_text"]:
            st.markdown(user_msg["display_text"])

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
        except Exception as exc:
            st.error(f"Request failed: `{exc}`")
