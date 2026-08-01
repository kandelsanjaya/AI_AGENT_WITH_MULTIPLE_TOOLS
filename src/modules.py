"""
modules.py
==========
All 12 feature-module UI implementations for EduSphere AI.

Each module is a standalone function. All modules use the refactored
utility helpers (guardrails, GROQ client, RAG pipeline, etc.).

Module 12 (Resume Builder) is new — generates professional resumes
via LLM and exports them as downloadable PDFs.
"""

from __future__ import annotations

import datetime
import io
import logging
from typing import Optional

import streamlit as st
from PIL import Image

from .config import CRISIS_RESOURCES
from .rag import VectorIndex, VectorStoreManager, extract_pdf_chunks, load_embedding_model
from .utils import (
    evaluate_guardrails,
    fetch_website_content,
    get_download_link,
    get_pdf_download_link,
    get_audio_download_link,
    groq_chat,
    stream_to_ui,
    duckduckgo_search,
    detect_greeting,
    get_greeting_response,
    search_lict_knowledge,
    log_interaction_to_json,
    logo_spinner,
)

log = logging.getLogger("edusphere.modules")

# Shared RAG manager (model is already cached by Streamlit)
_rag_manager = VectorStoreManager(load_embedding_model())
# ==============================================================================
# Shared helpers
# ==============================================================================

def _card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="g-card" style="{extra_style}">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_history() -> None:
    """Render the conversation history as a messenger-style chat using st.chat_message."""
    history = st.session_state.get("chat_history", [])
    if not history:
        st.markdown(
            '<div style="text-align:center;color:#475569;margin-top:80px;font-size:0.95rem;">'
            '🎓 Ask me anything — type below or use the mic</div>',
            unsafe_allow_html=True,
        )
        return
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        avatar = "👤" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["msg"])


def _render_export_buttons(content: str, base_name: str = "output") -> None:
    """Render a file type dropdown and download button for exporting content."""
    _card_open("margin-top: 10px;")
    st.markdown("**💾 Export Output**")
    col_sel, col_btn = st.columns([2, 1], vertical_alignment="bottom")
    with col_sel:
        import hashlib
        key_hash = hashlib.md5(f"{base_name}_{content[:50]}".encode("utf-8")).hexdigest()
        format_choice = st.selectbox(
            "Select Format",
            ["Plain Text (.txt)", "PDF Document (.pdf)", "Audio Podcast (.mp3)"],
            key=f"fmt_choice_{key_hash}"
        )
    with col_btn:
        if format_choice == "Plain Text (.txt)":
            st.download_button(
                label="⬇️ Download",
                data=content,
                file_name=f"{base_name}.txt",
                mime="text/plain",
                key=f"dl_txt_{key_hash}",
                use_container_width=True
            )
        elif format_choice == "PDF Document (.pdf)":
            from .utils import generate_pdf_bytes
            pdf_bytes = generate_pdf_bytes(content)
            st.download_button(
                label="⬇️ Download",
                data=pdf_bytes,
                file_name=f"{base_name}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{key_hash}",
                use_container_width=True
            )
        elif format_choice == "Audio Podcast (.mp3)":
            from .utils import generate_audio_bytes
            audio_bytes = generate_audio_bytes(content[:2000])
            if audio_bytes:
                st.download_button(
                    label="⬇️ Download",
                    data=audio_bytes,
                    file_name=f"{base_name}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_mp3_{key_hash}",
                    use_container_width=True
                )
            else:
                st.info("🔇 MP3 requires gTTS", icon="🔇")
    _card_close()




# ==============================================================================
# MODULE 0 — Dashboard (Lucy-AI style workspace with EduSphere cards)
# ==============================================================================

def render_dashboard() -> None:
    """🏠 EduSphere AI Dashboard — Lucy-AI style workspace home."""
    user = st.session_state.user_info or {}
    name = user.get("name", "User")
    role = user.get("role", "Student")
    now = datetime.datetime.now()
    hour = now.hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    total_q = st.session_state.get("total_queries", 0)
    chat_len = len(st.session_state.get("chat_history", []))
    session_mins = int((now - st.session_state.get("session_start", now)).total_seconds() / 60)

    # ── Stat Row ──
    st.markdown(
        f"""
        <div style="margin-bottom:20px;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700;
                        color:var(--text); margin-bottom:18px;">
                Your AI Agent Workspace
                <span style="font-size:0.95rem; color:var(--accent); margin-left:8px;">+</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px;">
                <div style="background:var(--card); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:18px 20px;">
                    <div style="font-size:0.68rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase;
                                margin-bottom:8px;">Total Queries</div>
                    <div style="font-size:2rem; font-weight:800; color:var(--text); font-family:'Orbitron',monospace;">{total_q}</div>
                    <div style="font-size:0.72rem; color:#4ade80; margin-top:6px;">▲ Active session</div>
                </div>
                <div style="background:var(--card); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:18px 20px;">
                    <div style="font-size:0.68rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase;
                                margin-bottom:8px;">Chat Messages</div>
                    <div style="font-size:2rem; font-weight:800; color:var(--text); font-family:'Orbitron',monospace;">{chat_len}</div>
                    <div style="font-size:0.72rem; color:var(--accent); margin-top:6px;">▲ This session</div>
                </div>
                <div style="background:var(--card); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:18px 20px;">
                    <div style="font-size:0.68rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase;
                                margin-bottom:8px;">Session Duration</div>
                    <div style="font-size:2rem; font-weight:800; color:var(--text); font-family:'Orbitron',monospace;">{session_mins}m</div>
                    <div style="font-size:0.72rem; color:var(--accent2); margin-top:6px;">Since login</div>
                </div>
                <div style="background:var(--card); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:18px 20px;">
                    <div style="font-size:0.68rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase;
                                margin-bottom:8px;">Role Access</div>
                    <div style="font-size:1.4rem; font-weight:800; color:var(--text); font-family:'Orbitron',monospace;">{role[:5].upper()}</div>
                    <div style="font-size:0.72rem; color:#fb923c; margin-top:6px;">● Authenticated</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Main area + right panel ──
    left_col, right_col = st.columns([2.2, 1], gap="medium")

    with left_col:
        # Greeting card
        _card_open()
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="font-size:1.5rem; font-weight:700; color:var(--text); margin-bottom:4px;">
                        {greeting}, <span style="color:var(--accent);">{name}</span>! 👋
                    </div>
                    <div style="color:var(--sub); font-size:0.85rem;">
                        How can EduSphere AI help you today?
                    </div>
                </div>
                <div style="width:90px; height:90px; border-radius:50%;
                            background:radial-gradient(circle at 35% 35%, #8b5cf6 0%, #1e1b4b 60%, #0a0b16 100%);
                            box-shadow:0 0 30px rgba(139,92,246,0.5); display:flex; align-items:center;
                            justify-content:center; font-size:2.2rem; flex-shrink:0;">
                    🎓
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        _card_close()

        # Quick ask
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### 💬 Quick Ask EduSphere AI")
        quick_q = st.text_area(
            "Ask me anything...",
            placeholder="Ask me anything — e.g. 'Explain Newton's 3rd law', 'Write a Python function...'",
            height=80,
            key="dashboard_quick_ask",
            label_visibility="collapsed"
        )
        qcol1, qcol2 = st.columns([1, 4])
        with qcol1:
            ask_btn = st.button("▶ Ask", key="dashboard_ask_btn", use_container_width=True)
        with qcol2:
            st.caption("Powered by Groq LLaMA 3.1 · RAG Studio")

        if ask_btn and quick_q.strip():
            with logo_spinner("Thinking..."):
                answer = groq_chat(
                    quick_q.strip(),
                    system="You are EduSphere AI, a helpful educational assistant. Give a concise but thorough answer."
                )
                st.session_state.total_queries = total_q + 1
            st.markdown(f"**🤖 EduSphere AI:**\n\n{answer}")
        _card_close()

        # Quick-access module cards
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🚀 Quick Module Access")
        modules_grid = [
            ("📝", "Summarise PDF", "📝 Executive Summariser", "#00f0ff"),
            ("🎨", "Generate Image", "🎨 AI Image Generator", "#8b5cf6"),
            ("💻", "Write Code", "💻 Code Lab & Explainer", "#4ade80"),
            ("🌍", "Translate Text", "🌍 Academic Translator", "#fb923c"),
            ("📊", "Analyse Data", "📊 System Analytics", "#f472b6"),
            ("🧪", "Take Quiz", "🧪 Quiz & Assessment Generator", "#38bdf8"),
        ]
        mcols = st.columns(3)
        for idx, (icon, label, nav_key, color) in enumerate(modules_grid):
            with mcols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                border-radius:12px; padding:14px 12px; text-align:center; margin-bottom:10px;
                                transition:all 0.2s; cursor:pointer;">
                        <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
                        <div style="font-size:0.78rem; font-weight:600; color:var(--text);">{label}</div>
                        <div style="margin-top:8px;">
                            <span style="font-size:0.65rem; background:rgba(255,255,255,0.06);
                                         padding:2px 8px; border-radius:6px; color:{color};">Open</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with right_col:
        # Memory / Vector Index Status
        _card_open()
        st.markdown("#### 🗃️ Memory Status")
        vi = st.session_state.get("vector_index")
        doc_count = len(vi.texts) if vi else 0
        mem_pct = min(int(doc_count / 2), 100)
        st.markdown(
            f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                            color:var(--sub); margin-bottom:6px;">
                    <span>Memory Usage</span>
                    <span style="color:{'#4ade80' if mem_pct < 60 else '#fb923c'};">
                        {'Good' if mem_pct < 60 else 'Moderate'}
                    </span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:99px; height:6px;">
                    <div style="background:linear-gradient(90deg,var(--accent),var(--accent2));
                                width:{max(mem_pct,3)}%; height:6px; border-radius:99px;"></div>
                </div>
                <div style="font-size:0.7rem; color:var(--sub); margin-top:5px;">{doc_count} / 200 items</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        _card_close()

        # Recent chat activity
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### 💬 Recent Chats")
        history = st.session_state.get("chat_history", [])
        if history:
            recents = [m for m in history if m["role"] == "user"][-4:][::-1]
            for msg in recents:
                preview = msg["msg"][:42] + "..." if len(msg["msg"]) > 42 else msg["msg"]
                st.markdown(
                    f"""
                    <div style="padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.05);
                                font-size:0.78rem; color:var(--sub); display:flex; justify-content:space-between;">
                        <span style="color:var(--text);">💬 {preview}</span>
                        <span style="font-size:0.68rem; flex-shrink:0; margin-left:8px;">now</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No chat history yet — start a conversation in EduChat!")
        _card_close()

        # Model selector (display only)
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### 🤖 Active Model")
        from .config import AVAILABLE_MODELS
        model_options = list(AVAILABLE_MODELS.keys()) if isinstance(AVAILABLE_MODELS, dict) else AVAILABLE_MODELS
        if "dashboard_model" not in st.session_state:
            st.session_state.dashboard_model = model_options[0] if model_options else "llama-3.1-8b-instant"
        chosen = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.dashboard_model) if st.session_state.dashboard_model in model_options else 0,
            key="dashboard_model_sel",
            label_visibility="collapsed"
        )
        st.session_state.dashboard_model = chosen
        st.markdown(
            f'<div style="font-size:0.72rem; color:#4ade80; margin-top:4px;">● Online · Groq Accelerated</div>',
            unsafe_allow_html=True
        )
        _card_close()

        # EduSphere info card (kept!)
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### 🎓 EduSphere AI")
        st.markdown(
            """
            <div style="font-size:0.78rem; color:var(--sub); line-height:1.7;">
                <div style="color:var(--accent); font-weight:600; margin-bottom:6px;">
                    Enterprise Educational Ecosystem
                </div>
                <div>● 14 AI-powered modules</div>
                <div>● FAISS RAG document search</div>
                <div>● 7 premium UI themes</div>
                <div>● Groq LLaMA 3.1 inference</div>
                <div>● 3D CesiumJS globe</div>
                <div style="margin-top:8px; color:var(--accent2); font-size:0.7rem;">
                    Powered by Groq · FAISS · Streamlit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        _card_close()


# ==============================================================================
# MODULE 1 — RAG Chatbot (with LICT Knowledge Integration)
# ==============================================================================

def render_educhat() -> None:
    """🧠 EduChat & RAG Studio — document-aware conversational AI with LICT campus knowledge."""
    # Inject JavaScript listener in the parent window to handle voice prompts from the iframe
    st.markdown(
        """
        <script>
            if (!window.voicePromptListenerAdded) {
                window.addEventListener('message', function(event) {
                    if (event.data && event.data.type === 'voice_prompt') {
                        const text = event.data.text;
                        const chatInput = document.querySelector('textarea[data-testid="stChatInputTextArea"]') || document.querySelector('.stChatInput textarea');
                        if (chatInput) {
                            const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
                            nativeSetter.call(chatInput, text);
                            chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                            setTimeout(() => {
                                const sendBtn = chatInput.closest('[data-testid="stChatInput"]').querySelector('button');
                                if (sendBtn) {
                                    sendBtn.click();
                                }
                            }, 100);
                        }
                    }
                });
                window.voicePromptListenerAdded = true;
            }
        </script>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 EduChat & RAG Search Engine")

    col_chat, col_panel = st.columns([2, 1])

    # ── Right panel: document management ──────────────────────────────────────
    with col_panel:
        _card_open()
        st.markdown("**📄 Document Vectorisation Hub**")

        uploaded_pdf = st.file_uploader(
            "Upload Academic PDF",
            type=["pdf"],
            key="rag_pdf_uploader",
            help="Upload a PDF to enable RAG-powered answers.",
        )

        if uploaded_pdf:
            if st.button("⚡ Index Document", key="btn_index_doc"):
                with logo_spinner("Extracting & Embedding PDF chunks…"):
                    try:
                        chunks = extract_pdf_chunks(uploaded_pdf)
                        vi = _rag_manager.build_index(chunks)
                        st.session_state.vector_index = vi
                        st.success(f"✅ Indexed **{vi.size}** chunks into FAISS!")
                        log.info("Document indexed: %d chunks.", vi.size)
                    except Exception as exc:
                        st.error(f"❌ Indexing failed: {exc}")
                        log.error("Indexing error: %s", exc)

        vi: Optional[VectorIndex] = st.session_state.get("vector_index")
        if vi and vi.size > 0:
            st.info(f"📚 FAISS Active — **{vi.size}** chunks loaded.")

        if st.button("🗑️ Clear Chat History", key="btn_clear_history"):
            st.session_state.chat_history = []
            st.rerun()

        _card_close()

    # ── Left panel: chat interface ─────────────────────────────────────────────
    with col_chat:
        # Unified chat box container (outer card style)
        with st.container(border=True):
            # Scrollable chat history area (no border)
            chat_container = st.container(height=380, border=False)
            with chat_container:
                _render_chat_history()
    
            st.markdown("<hr style='margin: 8px 0; border-top: 1px solid rgba(255, 255, 255, 0.08); border-bottom: none;'>", unsafe_allow_html=True)
    
            # Also check for voice query param
            voice_param = st.query_params.get("voice_prompt", "")
            
            # Display transcribed audio notification if received
            if voice_param:
                st.info(f"🎙️ **Transcribed Audio:** {voice_param}")
                
            prompt = st.chat_input("💬 Ask EduSphere anything…", key="edu_chat_input")
            
            st.components.v1.html(
                """
                <style>
                    html, body {
                        margin: 0;
                        padding: 0;
                        background: transparent;
                        overflow: hidden;
                    }
                    .voice-wave {
                        display: inline-flex;
                        align-items: center;
                        gap: 3px;
                        height: 12px;
                        margin-left: 6px;
                    }
                    .voice-bar {
                        width: 2px;
                        height: 100%;
                        background: #ef4444;
                        border-radius: 1px;
                        animation: bounce 0.8s ease-in-out infinite alternate;
                    }
                    .voice-bar:nth-child(2) { animation-delay: 0.15s; }
                    .voice-bar:nth-child(3) { animation-delay: 0.3s; }
                    .voice-bar:nth-child(4) { animation-delay: 0.45s; }
                    .voice-bar:nth-child(5) { animation-delay: 0.6s; }

                    @keyframes bounce {
                        0% { transform: scaleY(0.3); }
                        100% { transform: scaleY(1.1); }
                    }
                </style>
                <div style="font-family:'Inter',sans-serif; display:flex; flex-direction:column;
                            background-color: rgb(38, 39, 48); border: 1px solid transparent; position: relative;
                            padding: 0.4rem 0.6rem; border-radius: 0.5rem; box-sizing: border-box; width: 100%; margin: 0 auto;">
                    <div style="display:flex; flex-direction:row; align-items:center; gap:0.5rem;">
                        <button type="button" id="mic-btn"
                            style="background:transparent;border:none;font-size:1.15rem;color:#ececf1;
                                   cursor:pointer;outline:none;padding:0;transition:all .2s;display:flex;align-items:center;">🎙️</button>
                        <select id="lang-select"
                            style="background:transparent;color:#ececf1;border:none;
                                   font-size:0.8rem;outline:none;cursor:pointer;padding:1px 2px; width:45px;">
                            <option value="en-US" style="background:#262730;color:#ececf1;">EN</option>
                            <option value="ne-NP" style="background:#262730;color:#ececf1;">NE</option>
                            <option value="hi-IN" style="background:#262730;color:#ececf1;">HI</option>
                            <option value="es-ES" style="background:#262730;color:#ececf1;">ES</option>
                            <option value="fr-FR" style="background:#262730;color:#ececf1;">FR</option>
                        </select>
                        <span id="listening-indicator" style="color: #ef4444; font-size: 0.75rem; font-style: italic; display: none; align-items:center; gap:4px;">
                            🔴 Listening
                            <span class="voice-wave">
                                <span class="voice-bar"></span>
                                <span class="voice-bar"></span>
                                <span class="voice-bar"></span>
                                <span class="voice-bar"></span>
                                <span class="voice-bar"></span>
                            </span>
                        </span>
                    </div>
                    
                    <textarea id="live-transcription-box" placeholder="Transcribed text will appear here..." style="width: 100%; height: 32px; background: rgba(0,0,0,0.25); color: #ececf1; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 4px 6px; font-size: 0.85rem; outline: none; resize: none; margin-top: 4px; box-sizing: border-box; font-family: inherit;"></textarea>
                    
                    <div style="display: flex; gap: 8px; margin-top: 4px; justify-content: flex-end;">
                        <button type="button" id="copy-btn" style="background: rgba(255,255,255,0.06); color: #ececf1; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 2px 6px; font-size: 0.72rem; cursor: pointer; outline: none;">📋 Copy</button>
                        <button type="button" id="ask-btn" style="background: #00f0ff; color: #0f172a; border: none; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: bold; cursor: pointer; outline: none;">💬 Ask Bot</button>
                    </div>
                </div>
                <script>
                    const micBtn = document.getElementById('mic-btn');
                    const langSel = document.getElementById('lang-select');
                    const indicator = document.getElementById('listening-indicator');
                    const textBox = document.getElementById('live-transcription-box');
                    const copyBtn = document.getElementById('copy-btn');
                    const askBtn = document.getElementById('ask-btn');
            
                    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (!SR) {
                        micBtn.style.opacity = '0.3';
                        micBtn.title = 'Speech recognition not supported';
                    } else {
                        const rec = new SR();
                        rec.continuous = false;
                        rec.interimResults = true;
            
                        micBtn.addEventListener('click', () => {
                            rec.lang = langSel.value;
                            rec.start();
                            micBtn.style.color = '#ef4444';
                            micBtn.style.transform = 'scale(1.2)';
                            indicator.style.display = 'inline-flex';
                            textBox.value = '';
                        });
            
                        rec.onresult = (e) => {
                            let text = '';
                            for (let i = e.resultIndex; i < e.results.length; ++i) {
                                text += e.results[i][0].transcript;
                            }
                            textBox.value = text;
                        };
            
                        rec.onspeechend = () => rec.stop();
                        rec.onerror = (e) => {
                            micBtn.style.color = '#ececf1';
                            micBtn.style.transform = 'none';
                            indicator.style.display = 'none';
                        };
                        rec.onend = () => {
                            micBtn.style.color = '#ececf1';
                            micBtn.style.transform = 'none';
                            indicator.style.display = 'none';
                        };
                    }

                    copyBtn.addEventListener('click', () => {
                        textBox.select();
                        navigator.clipboard.writeText(textBox.value).then(() => {
                            copyBtn.textContent = '✅ Copied!';
                            setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
                        }).catch(() => {
                            // Fallback
                            document.execCommand('copy');
                            copyBtn.textContent = '✅ Copied!';
                            setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
                        });
                    });

                    askBtn.addEventListener('click', () => {
                        const text = textBox.value.trim();
                        if (text) {
                            try {
                                const parentDoc = window.parent.document;
                                const chatInputContainer = parentDoc.querySelector('div[data-testid="stChatInput"]') || parentDoc.querySelector('.stChatInput');
                                if (chatInputContainer) {
                                    const chatInput = chatInputContainer.querySelector('textarea, input');
                                    if (chatInput) {
                                        const proto = chatInput.tagName === 'TEXTAREA' ? window.parent.HTMLTextAreaElement.prototype : window.parent.HTMLInputElement.prototype;
                                        const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                                        nativeSetter.call(chatInput, text);
                                        chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                                        setTimeout(() => {
                                            const sendBtn = chatInputContainer.querySelector('button');
                                            if (sendBtn) {
                                                sendBtn.click();
                                            }
                                        }, 100);
                                        textBox.value = '';
                                    }
                                } else {
                                    fallbackRedirect(text);
                                }
                            } catch (e) {
                                fallbackRedirect(text);
                            }
                        }
                    });

                    function fallbackRedirect(text) {
                        let parentUrl = document.referrer;
                        if (!parentUrl) {
                            try {
                                parentUrl = window.parent.location.href;
                            } catch (e) {
                                parentUrl = window.location.origin;
                            }
                        }
                        const url = new URL(parentUrl);
                        url.searchParams.set('voice_prompt', text);
                        window.parent.location = url.toString();
                        textBox.value = '';
                    }
                </script>
                """,
                height=100,
            )

        # Trigger chat processing if voice prompt was received
        if voice_param and not prompt:
            prompt = voice_param

        if not prompt:
            return

        st.session_state.total_queries = st.session_state.get("total_queries", 0) + 1
        now_time = datetime.datetime.now().strftime("%H:%M")

        # ── Smart greeting detection (LICT Campus greetings) ─────────────────
        greeting_key = detect_greeting(prompt)
        if greeting_key:
            greeting_resp = get_greeting_response(greeting_key)
            st.session_state.chat_history.append({"role": "user", "msg": prompt, "time": now_time})
            st.session_state.chat_history.append({"role": "assistant", "msg": greeting_resp, "time": now_time})
            log_interaction_to_json(
                st.session_state.get("user_info", {}).get("name", "user"),
                "user", prompt,
            )
            log_interaction_to_json(
                st.session_state.get("user_info", {}).get("name", "user"),
                "assistant", greeting_resp,
            )
            if "voice_prompt" in st.query_params:
                st.query_params.clear()
            st.rerun()

        # Guardrails
        guards = evaluate_guardrails(prompt)
        if guards.crisis:
            st.session_state.blocked_count = st.session_state.get("blocked_count", 0) + 1
            st.markdown(
                f'<div class="g-card" style="border-color:#ef4444;">{CRISIS_RESOURCES}</div>',
                unsafe_allow_html=True,
            )
            if "voice_prompt" in st.query_params:
                st.query_params.clear()
            st.rerun()
        if guards.harmful or guards.private:
            st.session_state.blocked_count = st.session_state.get("blocked_count", 0) + 1
            st.error("🚫 Prompt blocked: Violates System Security or Data Privacy Policy.")
            if "voice_prompt" in st.query_params:
                st.query_params.clear()
            st.rerun()

        # Record user message
        st.session_state.chat_history.append(
            {"role": "user", "msg": prompt, "time": now_time}
        )

        # ── LICT Knowledge Base context injection ────────────────────────────
        context = ""
        used_web_search = False
        lict_context = search_lict_knowledge(prompt)

        # RAG context retrieval (document-based)
        if vi and vi.size > 0:
            try:
                context = _rag_manager.similarity_search(prompt, vi)
            except Exception as exc:
                st.warning(f"⚠️ RAG retrieval failed, answering without context: {exc}")
                log.warning("RAG search error: %s", exc)

        # Prepend College knowledge if relevant
        if lict_context:
            context = f"\n\n🏫 **College Knowledge Base:**\n{lict_context}\n\n{context}"

        is_personal_query = any(phrase in prompt.lower() for phrase in ["my name", "who am i", "do you know me", "who i am"])

        if not context and not is_personal_query:
            # Fallback to web search
            with logo_spinner("🔍 Context not found in local documents. Searching the web..."):
                web_results = duckduckgo_search(prompt)
                if web_results:
                    used_web_search = True
                    web_context_pieces = []
                    for idx, res in enumerate(web_results, 1):
                        web_context_pieces.append(
                            f"Source [{idx}]: {res['title']}\n"
                            f"URL: {res['link']}\n"
                            f"Snippet: {res['snippet']}"
                        )
                    context = "\n\n📄 **Live Web Search Results:**\n" + "\n\n".join(web_context_pieces) + "\n"

        full_prompt = f"{context}\n\nUser Question: {prompt}" if context else prompt

        # Get user profile information for personalization (excluding sensitive details like email or ID)
        user_info = st.session_state.get("user_info", {})
        user_name = user_info.get("name", "User")
        user_role = user_info.get("role", "Student")

        custom_ai_name = st.session_state.get("custom_ai_name", "EduSphere AI")

        identity_context = (
            f"\n\nYou are talking to a user named '{user_name}' who holds the role '{user_role}'. "
            f"Your name is currently '{custom_ai_name}'. "
            "If they ask who you are or what your name is, state that your name is 'EduSphere AI' (or your custom name if they previously assigned one), and tell them 'you can call me anything you'd like'. "
            "If the user gives you a new name (e.g. saying 'I will call you Jarvis' or 'your name is now Jarvis'), you must happily accept it, and you MUST include the tag `[SET_AI_NAME: <NewName>]` (for example: `[SET_AI_NAME: Jarvis]`) anywhere in your reply so the system remembers it. "
            "If they ask who they are, what their name is, or if you know them, answer them naturally like a human "
            "using this information. Do NOT under any circumstances disclose any other personal details like their email, "
            "password hash, or user ID."
        )

        if used_web_search:
            system_role = (
                "You are an expert, encouraging academic AI tutor and College AI Assistant. "
                "Answer questions precisely based on the Live Web Search Results context provided. "
                "Cite your web sources (e.g. Source [1], Source [2], etc.) and links in the response. "
                "CRITICAL: You must write your entire explanation and answer in the same language as the user's question."
                f"{identity_context}"
            )
        else:
            system_role = (
                "You are an expert, encouraging academic AI tutor and College AI Assistant. "
                "Answer questions precisely based on context provided if available. "
                "If college knowledge context is provided, ground your answer in it and mention it naturally. "
                "If context is provided, cite relevant sections. "
                "If not, answer from your training knowledge. "
                "CRITICAL: You must write your entire explanation and answer in the same language as the user's question."
                f"{identity_context}"
            )

        # Collect response and store, then rerun to show inside the scrollable container
        with logo_spinner("🤖 EduSphere is thinking…"):
            response_text = groq_chat(full_prompt, system=system_role)

        # Parse custom name request from response if present
        import re
        name_match = re.search(r'\[SET_AI_NAME:\s*([^\]]+)\]', response_text)
        if name_match:
            st.session_state.custom_ai_name = name_match.group(1).strip()
            response_text = re.sub(r'\[SET_AI_NAME:\s*([^\]]+)\]', '', response_text).strip()

        st.session_state.chat_history.append(
            {"role": "assistant", "msg": response_text, "time": now_time}
        )

        # Log interaction
        log_interaction_to_json(
            st.session_state.get("user_info", {}).get("name", "user"),
            "user", prompt,
        )
        log_interaction_to_json(
            st.session_state.get("user_info", {}).get("name", "user"),
            "assistant", response_text,
        )

        if "voice_prompt" in st.query_params:
            st.query_params.clear()
        st.rerun()


# ==============================================================================
# MODULE 2 — Study Planner
# ==============================================================================

def render_study_planner() -> None:
    """📚 Automated Study Plan Generator."""
    st.markdown("### 📚 Automated Study Plan Generator")
    col1, col2 = st.columns([1, 1])

    with col1:
        _card_open()
        topic = st.text_input("Target Subject / Course", "Quantum Mechanics & Computing")
        days = st.slider("Preparation Duration (Days)", 3, 30, 7)
        hours_per_day = st.number_input("Study Hours / Day", min_value=1, max_value=12, value=4)
        level = st.selectbox("Current Knowledge Level", ["Beginner", "Intermediate", "Advanced"])
        _card_close()

    with col2:
        if st.button("🚀 Generate Study Roadmap", key="btn_study_plan"):
            if not topic.strip():
                st.warning("Please enter a topic.")
                return
            prompt = (
                f"Generate a structured {days}-day study plan for '{topic}'. "
                f"Daily budget: {hours_per_day} hours. Target level: {level}. "
                "Break into daily topics, key resources, and revision checkpoints. "
                "Use clear headings and bullet points."
            )
            with logo_spinner("Synthesising optimal study plan…"):
                result = groq_chat(prompt, system="You are an expert curriculum designer.")

            _card_open()
            st.markdown(result)
            _card_close()
            _render_export_buttons(result, "study_plan")


# ==============================================================================
# MODULE 3 — Socratic Clarifier
# ==============================================================================

def render_socratic_clarifier() -> None:
    """🔬 Socratic Concept Deconstructor."""
    st.markdown("### 🔬 Socratic Concept Deconstructor")
    st.write("Deconstruct complex theories through guided Socratic questioning.")

    concept = st.text_input("Enter Concept to Master", "Fourier Transforms")

    if st.button("Deconstruct Concept", key="btn_socratic"):
        if not concept.strip():
            st.warning("Please enter a concept.")
            return
        prompt = (
            f"Explain '{concept}' using the Socratic method:\n"
            "1. **Core Intuition** – provide a real-world analogy.\n"
            "2. **Mathematical / Logical Breakdown** – step-by-step derivation.\n"
            "3. **Three Socratic Probing Questions** – to test deep comprehension.\n"
            "4. **Common Misconceptions** – pitfalls learners should avoid."
        )
        with logo_spinner("Deconstructing concept…"):
            result = groq_chat(prompt, system="You are a Socratic Academic Mentor.")
        _card_open()
        st.markdown(result)
        _card_close()
        _render_export_buttons(result, "socratic_analysis")


# ==============================================================================
# MODULE 4 — Quiz Generator
# ==============================================================================

def render_quiz_generator() -> None:
    """🧪 Automated Quiz & Flashcard Engine."""
    st.markdown("### 🧪 Automated Quiz & Flashcard Engine")
    col1, col2 = st.columns([1, 1])

    with col1:
        _card_open()
        quiz_topic = st.text_input("Quiz Subject Matter", "Cellular Biology & ATP Synthesis")
        num_q = st.slider("Number of Questions", 3, 15, 5)
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["High School", "Undergraduate", "Postgraduate"],
            value="Undergraduate",
        )
        q_type = st.selectbox("Question Type", ["Multiple Choice (A–D)", "True / False", "Short Answer"])
        _card_close()

    with col2:
        if st.button("🎯 Generate Assessment", key="btn_quiz"):
            if not quiz_topic.strip():
                st.warning("Please enter a subject matter.")
                return
            prompt = (
                f"Create a {num_q}-question {q_type} quiz on '{quiz_topic}' "
                f"for {difficulty} level students. "
                "For multiple choice, provide choices A–D. "
                "Highlight correct answers clearly and add detailed explanations for each."
            )
            with logo_spinner("Formulating questions…"):
                result = groq_chat(prompt)
            _card_open()
            st.markdown(result)
            _card_close()
            _render_export_buttons(result, "quiz")


# ==============================================================================
# MODULE 5 — Code Lab
# ==============================================================================

def render_code_lab() -> None:
    """💻 Intelligent Code Debugger & Explainer."""
    st.markdown("### 💻 Intelligent Code Debugger & Explainer")

    code_snippet = st.text_area(
        "Paste Code Snippet",
        value=(
            "def binary_search(arr, target):\n"
            "    # Bug: high should be len(arr) - 1\n"
            "    low, high = 0, len(arr)\n"
            "    while low < high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: low = mid   # Bug: should be mid + 1\n"
            "        else: high = mid\n"
            "    return -1"
        ),
        height=180,
    )

    lang = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"])
    task = st.selectbox(
        "Analysis Task",
        [
            "Explain Line-by-Line",
            "Find & Fix All Bugs",
            "Optimise Time & Space Complexity",
            f"Translate to C++",
            "Write Unit Tests",
            "Add Type Hints / Annotations",
        ],
    )

    if st.button("⚡ Execute Analysis", key="btn_code_lab"):
        if not code_snippet.strip():
            st.warning("Please paste a code snippet.")
            return
        prompt = (
            f"Task: {task}\n\n"
            f"Language: {lang}\n\n"
            f"Code:\n```{lang.lower()}\n{code_snippet}\n```\n\n"
            "Provide a thorough, structured analysis with corrected code where applicable."
        )
        with logo_spinner("Analysing code structure…"):
            result = groq_chat(prompt, system="You are an expert Software Engineer & CS Professor.")
        _card_open()
        st.markdown(result)
        _card_close()
        _render_export_buttons(result, "code_analysis")


# ==============================================================================
# MODULE 6 — Academic Translator
# ==============================================================================

def render_translator() -> None:
    """🌍 Multi-Lingual Academic Translator."""
    st.markdown("### 🌍 Multi-Lingual Academic Translator")

    # ── Voice Input handling for Translator ──
    translate_voice = st.query_params.get("translate_voice_prompt", "")
    default_text = "Neural networks utilise backpropagation to update weights based on gradient loss."
    should_auto_translate = False
    
    if "translate_input" not in st.session_state:
        st.session_state.translate_input = default_text
        
    if translate_voice:
        st.session_state.translate_input = translate_voice
        should_auto_translate = True
        st.query_params.clear()

    # ── Translator Voice Input component ──
    st.components.v1.html(
        """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
            }
            .voice-wave {
                display: inline-flex;
                align-items: center;
                gap: 3px;
                height: 12px;
                margin-left: 6px;
            }
            .voice-bar {
                width: 2px;
                height: 100%;
                background: #ef4444;
                border-radius: 1px;
                animation: bounce 0.8s ease-in-out infinite alternate;
            }
            .voice-bar:nth-child(2) { animation-delay: 0.15s; }
            .voice-bar:nth-child(3) { animation-delay: 0.3s; }
            .voice-bar:nth-child(4) { animation-delay: 0.45s; }
            .voice-bar:nth-child(5) { animation-delay: 0.6s; }

            @keyframes bounce {
                0% { transform: scaleY(0.3); }
                100% { transform: scaleY(1.1); }
            }
        </style>
        <div style="font-family:'Inter',sans-serif; display:flex; flex-direction:column;
                    background-color: rgb(38, 39, 48); border: 1px solid transparent; position: relative;
                    padding: 0.4rem 0.6rem; border-radius: 0.5rem; box-sizing: border-box; width: 100%; margin-bottom: 6px;">
            <div style="display:flex; flex-direction:row; align-items:center; gap:0.5rem;">
                <button type="button" id="trans-mic-btn"
                    style="background:transparent;border:none;font-size:1.15rem;color:#ececf1;
                           cursor:pointer;outline:none;padding:0;transition:all .2s;display:flex;align-items:center;gap:6px;">🎙️ Speak to Translate</button>
                <select id="trans-lang-select"
                    style="background:transparent;color:#ececf1;border:none;
                           font-size:0.8rem;outline:none;cursor:pointer;padding:1px 2px; width:45px;">
                    <option value="en-US" style="background:#262730;color:#ececf1;">EN</option>
                    <option value="ne-NP" style="background:#262730;color:#ececf1;">NE</option>
                    <option value="hi-IN" style="background:#262730;color:#ececf1;">HI</option>
                    <option value="es-ES" style="background:#262730;color:#ececf1;">ES</option>
                    <option value="fr-FR" style="background:#262730;color:#ececf1;">FR</option>
                </select>
                <span id="trans-indicator" style="color: #ef4444; font-size: 0.75rem; font-style: italic; display: none; align-items:center; gap:4px;">
                    🔴 Listening
                    <span class="voice-wave">
                        <span class="voice-bar"></span>
                        <span class="voice-bar"></span>
                        <span class="voice-bar"></span>
                        <span class="voice-bar"></span>
                        <span class="voice-bar"></span>
                    </span>
                </span>
            </div>
            
            <textarea id="trans-transcription-box" placeholder="Transcribed text will appear here..." style="width: 100%; height: 32px; background: rgba(0,0,0,0.25); color: #ececf1; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 4px 6px; font-size: 0.85rem; outline: none; resize: none; margin-top: 4px; box-sizing: border-box; font-family: inherit;"></textarea>
            
            <div style="display: flex; gap: 8px; margin-top: 4px; justify-content: flex-end;">
                <button type="button" id="trans-copy-btn" style="background: rgba(255,255,255,0.06); color: #ececf1; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 2px 6px; font-size: 0.72rem; cursor: pointer; outline: none;">📋 Copy</button>
                <button type="button" id="trans-submit-btn" style="background: #00f0ff; color: #0f172a; border: none; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: bold; cursor: pointer; outline: none;">🌐 Translate</button>
            </div>
        </div>
        <script>
            const micBtn = document.getElementById('trans-mic-btn');
            const langSel = document.getElementById('trans-lang-select');
            const indicator = document.getElementById('trans-indicator');
            const textBox = document.getElementById('trans-transcription-box');
            const copyBtn = document.getElementById('trans-copy-btn');
            const submitBtn = document.getElementById('trans-submit-btn');
    
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) {
                micBtn.style.opacity = '0.3';
                micBtn.title = 'Speech recognition not supported';
            } else {
                const rec = new SR();
                rec.continuous = false;
                rec.interimResults = true;
    
                micBtn.addEventListener('click', () => {
                    rec.lang = langSel.value;
                    rec.start();
                    micBtn.style.color = '#ef4444';
                    indicator.style.display = 'inline-flex';
                    textBox.value = '';
                });
    
                rec.onresult = (e) => {
                    let text = '';
                    for (let i = e.resultIndex; i < e.results.length; ++i) {
                        text += e.results[i][0].transcript;
                    }
                    textBox.value = text;
                };
    
                rec.onspeechend = () => rec.stop();
                rec.onerror = (e) => {
                    micBtn.style.color = '#ececf1';
                    indicator.style.display = 'none';
                };
                rec.onend = () => {
                    micBtn.style.color = '#ececf1';
                    indicator.style.display = 'none';
                };
            }

            copyBtn.addEventListener('click', () => {
                textBox.select();
                navigator.clipboard.writeText(textBox.value).then(() => {
                    copyBtn.textContent = '✅ Copied!';
                    setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
                }).catch(() => {
                    document.execCommand('copy');
                    copyBtn.textContent = '✅ Copied!';
                    setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
                });
            });

            submitBtn.addEventListener('click', () => {
                const text = textBox.value.trim();
                if (text) {
                    try {
                        const parentDoc = window.parent.document;
                        const transInput = parentDoc.querySelector('textarea[aria-label="Source Text"]') || parentDoc.querySelector('.stTextArea textarea');
                        if (transInput) {
                            const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
                            nativeSetter.call(transInput, text);
                            transInput.dispatchEvent(new Event('input', { bubbles: true }));
                            setTimeout(() => {
                                const buttons = Array.from(parentDoc.querySelectorAll('button'));
                                const translateBtn = buttons.find(btn => btn.textContent.includes('Translate Text'));
                                if (translateBtn) {
                                    translateBtn.click();
                                }
                            }, 100);
                            textBox.value = '';
                        } else {
                            fallbackRedirect(text);
                        }
                    } catch (e) {
                        fallbackRedirect(text);
                    }
                }
            });

            function fallbackRedirect(text) {
                let parentUrl = document.referrer;
                if (!parentUrl) {
                    try {
                        parentUrl = window.parent.location.href;
                    } catch (e) {
                        parentUrl = window.location.origin;
                    }
                }
                const url = new URL(parentUrl);
                url.searchParams.set('translate_voice_prompt', text);
                window.parent.location = url.toString();
                textBox.value = '';
            }
        </script>
        """,
        height=100,
    )

    input_text = st.text_area(
        "Source Text",
        value=st.session_state.translate_input,
        height=120,
        key="translator_source_text"
    )
    
    # Sync typed text back to state
    st.session_state.translate_input = input_text

    col1, col2 = st.columns([1, 1])
    with col1:
        target_lang = st.selectbox(
            "Target Language",
            ["Nepali", "Spanish", "French", "German", "Chinese (Mandarin)", "Japanese", "Hindi", "Arabic"],
        )
    with col2:
        formality = st.selectbox("Formality Level", ["Academic / Formal", "Conversational", "Technical"])

    if st.button("🌐 Translate Text", key="btn_translate") or should_auto_translate:
        if not input_text.strip():
            st.warning("Please provide source text.")
            return
        prompt = (
            f"Translate the following academic text into {target_lang} "
            f"using a {formality} tone. Preserve specialised terminology. "
            "If any terms have no direct equivalent, provide a footnote explanation.\n\n"
            f"Text:\n{input_text}"
        )
        with logo_spinner("Translating…"):
            result = groq_chat(prompt)
        _card_open()
        st.subheader(f"Translation → {target_lang}")
        st.write(result)
        _card_close()
        _render_export_buttons(result, "translation")


# ==============================================================================
# MODULE 7 — Executive Summariser
# ==============================================================================

def render_summariser() -> None:
    """📝 Academic Text & Paper Summariser."""
    st.markdown("### 📝 Academic Text & Paper Summariser")

    long_text = st.text_area("Input Academic Passage", height=220)
    col1, col2 = st.columns([1, 1])
    with col1:
        format_type = st.selectbox(
            "Summary Format",
            ["Bullet Core Takeaways", "Executive One-Pager", "Abstract Style", "ELI5 (Simple Explanation)"],
        )
    with col2:
        word_limit = st.slider("Target Word Count", 50, 500, 150)

    if st.button("📑 Summarise", key="btn_summarise"):
        if not long_text.strip():
            st.warning("Please provide input text.")
            return
        prompt = (
            f"Summarise the following text using the '{format_type}' format. "
            f"Target approximately {word_limit} words. "
            "Be precise and preserve key academic insights.\n\n"
            f"Text:\n{long_text}"
        )
        with logo_spinner("Summarising…"):
            result = groq_chat(prompt)
        _card_open()
        st.markdown(result)
        _card_close()
        _render_export_buttons(result, "summary")


# ==============================================================================
# MODULE 8 — URL & Visual Intelligence
# ==============================================================================

def render_url_intelligence() -> None:
    """🖼️ Web & Image Scraping Engine."""
    st.markdown("### 🖼️ Web Content Intelligence Engine")

    target_url = st.text_input(
        "Web URL to Analyse",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    )

    if st.button("🔍 Analyse URL", key="btn_analyse_url"):
        if not target_url.strip():
            st.warning("Please enter a URL.")
            return
        with logo_spinner("Scraping webpage structure…"):
            data = fetch_website_content(target_url)

        if "error" in data:
            st.error(f"❌ Failed to extract content: {data['error']}")
            return

        _card_open()
        st.markdown(f"**🌐 Title:** {data['title']}")
        st.markdown(f"**🔗 URL:** {data['url']}")

        if data["headings"]:
            st.markdown("**📌 Headings Discovered:**")
            for h in data["headings"]:
                st.markdown(f"- {h}")

        if data["paragraphs"]:
            st.markdown("**📄 Key Paragraphs:**")
            for p in data["paragraphs"]:
                st.caption(f"• {p}")
        _card_close()

        combined_text = "\n".join(data["headings"] + data["paragraphs"])
        _render_export_buttons(combined_text, "url_analysis")

        if st.button("🤖 AI-Summarise This Page", key="btn_ai_summarise_url"):
            combined = "\n".join(data["paragraphs"])
            prompt = f"Summarise the key academic insights from this web content:\n\n{combined}"
            with logo_spinner("Generating AI summary…"):
                summary = groq_chat(prompt)
            _card_open()
            st.markdown("**🤖 AI Summary:**")
            st.markdown(summary)
            _card_close()
            _render_export_buttons(summary, "url_summary")


# ==============================================================================
# MODULE 9 — Background Remover
# ==============================================================================

def render_bg_remover() -> None:
    """🧹 Image Background Eraser."""
    st.markdown("### 🧹 AI-Powered Background Eraser")
    st.caption("Upload a JPG or PNG image to remove its background instantly.")

    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="bg_uploader")

    if not img_file:
        return

    try:
        img = Image.open(img_file).convert("RGBA")
    except Exception as exc:
        st.error(f"❌ Could not open image: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    if st.button("✂️ Remove Background", key="btn_remove_bg"):
        with logo_spinner("Processing image tensors…"):
            try:
                from rembg import remove
                output_img = remove(img)
            except Exception as exc:
                st.error(f"❌ Background removal failed: {exc}")
                log.error("rembg error: %s", exc)
                return

        with col2:
            st.image(output_img, caption="Background Removed", use_container_width=True)

        # Download as PNG
        buf_png = io.BytesIO()
        output_img.save(buf_png, format="PNG")
        st.download_button(
            "⬇️ Download PNG",
            data=buf_png.getvalue(),
            file_name="bg_removed.png",
            mime="image/png",
        )

        # Download as JPEG
        jpg_img = output_img.convert("RGB")
        buf_jpg = io.BytesIO()
        jpg_img.save(buf_jpg, format="JPEG", quality=95)
        st.download_button(
            "⬇️ Download JPEG",
            data=buf_jpg.getvalue(),
            file_name="bg_removed.jpg",
            mime="image/jpeg",
        )

        st.toast("✅ Background removed successfully!", icon="✂️")


# ==============================================================================
# MODULE 10 — Analytics
# ==============================================================================

def render_analytics() -> None:
    """📊 Platform Usage & Telemetry."""
    st.markdown("### 📊 Platform Usage & Telemetry Dashboard")

    vi = st.session_state.get("vector_index")
    faiss_size = vi.size if vi else 0

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔢 Total Queries", st.session_state.get("total_queries", 0))
    col_b.metric("🚫 Security Blocks", st.session_state.get("blocked_count", 0))
    col_c.metric("📚 FAISS Chunk Count", faiss_size)

    _card_open()
    st.markdown("**Session Metadata:**")
    session_start = st.session_state.get("session_start", datetime.datetime.now())
    elapsed = datetime.datetime.now() - session_start
    hours, rem = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(rem, 60)

    user_info = st.session_state.get("user_info", {})
    st.json(
        {
            "session_start": str(session_start),
            "session_duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "user_name": user_info.get("name", "—"),
            "user_role": user_info.get("role", "—"),
            "theme": st.session_state.get("theme", "—"),
            "rag_active": faiss_size > 0,
            "chat_messages": len(st.session_state.get("chat_history", [])),
        }
    )
    _card_close()


# ==============================================================================
# MODULE 11 — Architecture Blueprint
# ==============================================================================

def render_architecture() -> None:
    """🏛️ Executive Vision & System Architecture Documentation."""
    st.markdown("## 🏛️ Executive Vision & System Architecture")

    st.markdown("""
---
### Executive Vision & Core Goals

**EduSphere AI** is an enterprise-grade, privacy-first educational ecosystem capable of:
- **Adaptive Learning Paths** — personalised study plans and Socratic dialogue.
- **Real-Time RAG** — FAISS-backed context extraction from uploaded documents.
- **Multi-Modal Problem Solving** — code analysis, translation, summarisation, and image processing.
- **College AI Assistant Integration** — built-in knowledge base for college queries.
- **Professional Resume Builder** — AI-powered resume generation with PDF export.

---
### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER INTERFACE                   │
│            (Streamlit — Multi-Module SPA)           │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│         INPUT EVALUATION & GUARDRAIL ENGINE         │
│   (Regex Safety · Privacy · Crisis Classifier)      │
└──────────┬──────────────────────────────┬───────────┘
           │ BLOCKED                      │ SAFE
           ▼                              ▼
┌──────────────────┐           ┌──────────────────────┐
│  BLOCK & REPORT  │           │   ROUTING ENGINE     │
│ Crisis Resources │           │  (12 Feature Modules)│
└──────────────────┘           └────────┬─────────────┘
                                        │
                          ┌─────────────┴──────────────┐
                          │                            │
                    DIRECT LLM                    RAG MODE
                          │                            │
                          │               ┌────────────▼────────────┐
                          │               │  FAISS VECTOR INDEX     │
                          │               │  (SentenceTransformers) │
                          │               └────────────┬────────────┘
                          │                            │ Top-K Chunks
                          │               ┌────────────▼────────────┐
                          │               │  COLLEGE KNOWLEDGE BASE │
                          │               │  + CONTEXT AUGMENTER    │
                          │               └────────────┬────────────┘
                          │                            │
                          └──────────────┬─────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │  GROQ LLaMA-3.1 INFERENCE   │
                          │   (8B / 70B Instant API)    │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   STREAMING RESPONSE UI     │
                          │  + MULTI-FORMAT EXPORT       │
                          │  (TXT · PDF · MP3 · PNG)    │
                          └─────────────────────────────┘
```

---
### Security Architecture

| Layer | Mechanism |
|---|---|
| **Authentication** | bcrypt password hashing (cost=12) + SQLite persistence |
| **Input Safety** | Multi-pattern regex guardrails |
| **Privacy** | PII / credential pattern detection |
| **API Keys** | `.env` file, never logged or exposed |
| **Logging** | Structured — no sensitive data written |
| **Chat History** | SQLite with per-user session isolation |

---
### Database Schema

```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    full_name     TEXT,
    salt          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    theme         TEXT DEFAULT 'dark',
    created_at    TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    title         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE chats (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE document_embeddings (
    chunk_id       UUID PRIMARY KEY,
    session_id     UUID REFERENCES sessions(session_id),
    document_name  VARCHAR(255) NOT NULL,
    chunk_index    INT NOT NULL,
    chunk_content  TEXT NOT NULL,
    embedding_vec  VECTOR(384)  -- all-MiniLM-L6-v2 dimension
);
```

---
### Project Structure

```text
edusphere-ai/
├── .env                  # API keys (never commit)
├── DasaAI.py             # Streamlit entry point
├── data.json             # College knowledge base + greetings + interaction log
├── users.db              # SQLite user/session/chat store (auto-created)
├── src/
│   ├── __init__.py
│   ├── auth.py           # bcrypt authentication
│   ├── config.py         # constants & env loading (6 themes)
│   ├── database.py       # SQLite CRUD layer
│   ├── exceptions.py     # custom exception hierarchy
│   ├── main.py           # Streamlit entrypoint
│   ├── modules.py        # all 12 feature modules
│   ├── rag.py            # FAISS RAG pipeline
│   └── utils.py          # GROQ client, guardrails, exports & helpers
├── assets/
│   ├── styles.css        # premium CSS stylesheet
│   └── nepal_map.png     # login background
├── tests/
│   ├── test_auth.py
│   ├── test_rag.py
│   └── test_utils.py
├── logs/
│   └── app.log
└── requirements.txt
```
    """)


# ==============================================================================
# MODULE 12 — Resume Builder (NEW)
# ==============================================================================

def render_resume_builder() -> None:
    """📋 AI-Powered Professional Resume Builder."""
    st.markdown("### 📋 AI-Powered Professional Resume Builder")
    st.caption("Fill in your details below and let AI craft a professional resume for you.")

    _card_open()
    col1, col2 = st.columns(2)

    with col1:
        full_name = st.text_input("👤 Full Name", placeholder="Sanjaya Kandel", key="resume_name")
        email = st.text_input("📧 Email", placeholder="your@email.com", key="resume_email")
        phone = st.text_input("📱 Phone", placeholder="+977-9800000000", key="resume_phone")
        location = st.text_input("📍 Location", placeholder="Kathmandu, Nepal", key="resume_location")
        linkedin = st.text_input("🔗 LinkedIn / Portfolio URL", placeholder="https://linkedin.com/in/yourname", key="resume_linkedin")

    with col2:
        objective = st.text_area(
            "🎯 Career Objective / Summary",
            placeholder="Brief summary of your career goals and key strengths…",
            height=100,
            key="resume_objective",
        )
        education = st.text_area(
            "🎓 Education",
            placeholder="BSc. CSIT — Lumbini ICT Campus (2020-2024)\nGPA: 3.5/4.0",
            height=100,
            key="resume_education",
        )
        skills = st.text_input(
            "💡 Key Skills",
            placeholder="Python, JavaScript, React, Machine Learning, SQL",
            key="resume_skills",
        )

    experience = st.text_area(
        "💼 Work Experience",
        placeholder="Software Developer — TechCorp Nepal (2024-Present)\n"
        "• Built REST APIs using Django and PostgreSQL\n"
        "• Led a team of 3 junior developers",
        height=120,
        key="resume_experience",
    )

    projects = st.text_area(
        "🚀 Projects",
        placeholder="EduSphere AI — AI-powered educational platform\n"
        "• Built with Streamlit, GROQ API, FAISS\n"
        "• Features: RAG chatbot, quiz generator, resume builder",
        height=100,
        key="resume_projects",
    )

    certifications = st.text_input(
        "🏆 Certifications / Awards",
        placeholder="AWS Cloud Practitioner, Google IT Support Certificate",
        key="resume_certs",
    )

    resume_style = st.selectbox(
        "📐 Resume Style",
        ["Professional Modern", "Academic / Research", "Creative Tech", "Minimal Clean"],
        key="resume_style",
    )

    _card_close()

    col_gen, col_preview = st.columns([1, 2])

    with col_gen:
        generate_btn = st.button("✨ Generate Resume", key="btn_generate_resume", use_container_width=True)

    if generate_btn:
        if not full_name.strip():
            st.warning("Please enter at least your full name.")
            return

        prompt = f"""Generate a professional resume in clean markdown format using the following details:

**Full Name:** {full_name}
**Email:** {email}
**Phone:** {phone}
**Location:** {location}
**LinkedIn / Portfolio:** {linkedin}

**Career Objective:**
{objective}

**Education:**
{education}

**Skills:** {skills}

**Work Experience:**
{experience}

**Projects:**
{projects}

**Certifications / Awards:** {certifications}

**Style Preference:** {resume_style}

Please create a well-structured, professional resume with:
1. Clear section headings
2. Bullet points for experiences and projects
3. Professional language and action verbs
4. Proper formatting for ATS compatibility
5. A brief professional summary if objective is provided

Output in clean, well-formatted markdown."""

        with logo_spinner("✨ AI is crafting your resume…"):
            resume_text = groq_chat(
                prompt,
                system="You are a professional resume writer and career advisor. "
                       "Create polished, ATS-compatible resumes that highlight the candidate's strengths. "
                       "Use clean markdown formatting with clear sections.",
                max_tokens=3000,
            )

        st.session_state.generated_resume = resume_text

    # Display resume if generated
    if st.session_state.get("generated_resume"):
        resume_text = st.session_state.generated_resume

        with col_preview:
            _card_open()
            st.markdown("#### 📄 Your Generated Resume")
            st.markdown("---")
            st.markdown(resume_text)
            _card_close()

        _render_export_buttons(resume_text, f"resume_{full_name.replace(' ', '_')}")


# ==============================================================================
# MODULE 13 — AI Image Generator
# ==============================================================================

def render_image_generator() -> None:
    """🎨 AI Image Generator — Rewrites prompt via Groq and pulls image from Pollinations."""
    st.markdown("### 🎨 AI Image Generator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        _card_open()
        st.markdown("#### 📝 Describe Your Image Concept")
        
        user_prompt = st.text_area(
            "Enter image concept / description",
            placeholder="e.g. A futuristic learning university on Mars with students studying using neon hologram displays...",
            height=120,
            key="img_gen_prompt_input"
        )
        
        enhance_prompt = st.checkbox("✨ Enhance prompt with Groq AI (Recommended)", value=True)
        
        style_preset = st.selectbox(
            "🎨 Render Style Preset",
            [
                "📸 Realistic Photorealistic / Cinematic",
                "🎨 Digital Concept Art",
                "✏️ Anime & Illustration",
                "🖌️ Classical Oil Painting"
            ],
            index=0
        )
        
        aspect_ratio = st.selectbox(
            "📐 Aspect Ratio",
            ["1:1 (Square)", "16:9 (Widescreen)", "9:16 (Portrait)", "4:3 (Classic)"],
            index=1
        )
        
        gen_clicked = st.button("🎨 Generate Image", use_container_width=True)
        _card_close()
        
    with col2:
        _card_open()
        st.markdown("#### 🖼️ Image Output")
        
        if gen_clicked:
            if not user_prompt.strip():
                st.warning("⚠️ Please provide a description first.")
            else:
                enhanced = user_prompt
                if enhance_prompt:
                    with logo_spinner("Expanding and refining prompt via Groq..."):
                        style_instructions = {
                            "📸 Realistic Photorealistic / Cinematic": (
                                "Make the output a highly detailed photorealistic masterpiece. "
                                "Specify camera parameters like 'shot on 35mm lens, f/1.8, cinematic volumetric lighting, 8k resolution, sharp focus, realistic textures'."
                            ),
                            "🎨 Digital Concept Art": "Make the output a gorgeous digital concept art piece, with vibrant colors, dramatic lighting, and creative detailing.",
                            "✏️ Anime & Illustration": "Make the output a beautiful anime style illustration, clean line art, and soft cel shading.",
                            "🖌️ Classical Oil Painting": "Make the output look like a classical oil painting masterpiece, with detailed brush strokes and canvas texture."
                        }[style_preset]
                        
                        enhance_instructions = (
                            "You are a professional prompt engineer for Stable Diffusion/Midjourney. "
                            "Expand the user's short concept into a detailed visual prompt. "
                            f"Focus style directions: {style_instructions} "
                            "Keep your response strictly to the enhanced prompt text under 100 words. Do NOT include any intro or chat."
                        )
                        enhanced = groq_chat(user_prompt, system=enhance_instructions)
                else:
                    # Append default style suffix
                    style_suffix = {
                        "📸 Realistic Photorealistic / Cinematic": ", photorealistic, cinematic lighting, 8k resolution, shot on 35mm lens, highly detailed, sharp focus, masterpiece",
                        "🎨 Digital Concept Art": ", digital concept art, trending on artstation, detailed, vibrant colors",
                        "✏️ Anime & Illustration": ", anime illustration, clean lines, detailed graphic style",
                        "🖌️ Classical Oil Painting": ", classical oil painting, fine art, visible brush strokes, canvas texture"
                    }[style_preset]
                    enhanced = user_prompt + style_suffix
                
                # Aspect Ratio to Dimensions
                dim_map = {
                    "1:1 (Square)": (1024, 1024),
                    "16:9 (Widescreen)": (1344, 768),
                    "9:16 (Portrait)": (768, 1344),
                    "4:3 (Classic)": (1024, 768)
                }
                width, height = dim_map[aspect_ratio]
                
                # Build URL using Pollinations
                import urllib.parse
                safe_prompt = urllib.parse.quote(enhanced)
                import time
                seed = int(time.time())
                img_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&nologo=true&seed={seed}"
                
                with logo_spinner("Generating image via high-speed CDN..."):
                    # Render image
                    st.image(img_url, caption=f"Generated Image: {user_prompt}", use_container_width=True)
                    
                    try:
                        import requests
                        from io import BytesIO
                        from PIL import Image

                        resp = requests.get(img_url, timeout=20)
                        if resp.status_code == 200:
                            img_data = resp.content
                            img = Image.open(BytesIO(img_data))
                            
                            # Convert to PNG, JPG, WebP bytes
                            png_io = BytesIO()
                            img.save(png_io, format="PNG")
                            png_bytes = png_io.getvalue()
                            
                            jpg_io = BytesIO()
                            img.convert("RGB").save(jpg_io, format="JPEG")
                            jpg_bytes = jpg_io.getvalue()
                            
                            webp_io = BytesIO()
                            img.save(webp_io, format="WEBP")
                            webp_bytes = webp_io.getvalue()
                            
                            st.markdown("##### 📥 Download in Different Formats:")
                            col_d1, col_d2, col_d3 = st.columns(3)
                            with col_d1:
                                st.download_button("💾 Download .png", png_bytes, file_name="generated_image.png", mime="image/png", use_container_width=True)
                            with col_d2:
                                st.download_button("💾 Download .jpg", jpg_bytes, file_name="generated_image.jpg", mime="image/jpeg", use_container_width=True)
                            with col_d3:
                                st.download_button("💾 Download .webp", webp_bytes, file_name="generated_image.webp", mime="image/webp", use_container_width=True)
                        else:
                            st.error("Failed to download image bytes for format conversion.")
                    except Exception as e:
                        st.error(f"Error converting image formats: {e}")
        else:
            st.info("ℹ️ Enter a description and click Generate to see the image output here.")
        _card_close()


# ==============================================================================
# MODULE 14 — Interactive 3D Globe
# ==============================================================================

def render_globe_map() -> None:
    """🌍 Interactive 3D Globe Map for Academic Learning."""
    st.markdown("### 🌍 Interactive 3D Globe & Weather Intelligence")

    col1, col2 = st.columns([2, 1])

    # Predefined places of academic/scientific interest
    PREDEFINED_PLACES = {
        "Mount Everest": {"lat": 27.9881, "lng": 86.9250, "desc": "Highest point on Earth, located in the Himalayas on the border of Nepal and China."},
        "Great Pyramids of Giza": {"lat": 29.9792, "lng": 31.1342, "desc": "Ancient structures located near Cairo, Egypt, built as tombs for Pharaohs."},
        "Mariana Trench": {"lat": 11.3493, "lng": 142.1996, "desc": "Deepest known point in Earth's oceans, located in the Western Pacific."},
        "Amazon Rainforest": {"lat": -3.4653, "lng": -62.2159, "desc": "World's largest tropical rainforest, famous for its biodiverse ecosystem."},
        "CERN (Hadron Collider)": {"lat": 46.2333, "lng": 6.0491, "desc": "World's largest particle physics laboratory, located on the France-Switzerland border."}
    }

    with col1:
        _card_open()
        st.markdown("#### 🗺️ 3D Globe Viewer (Drag to rotate, scroll to zoom)")

        import json
        points_data = [
            {"lat": val["lat"], "lng": val["lng"], "name": name, "color": "red", "size": 0.5}
            for name, val in PREDEFINED_PLACES.items()
        ]

        html_code = f"""
        <link href="https://cesium.com/downloads/cesiumjs/releases/1.105/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                background: #000;
                font-family: 'Space Grotesk', sans-serif;
            }}
            #mapWrapper {{
                position: relative;
                width: 100%;
                height: 500px;
                border-radius: 8px;
                overflow: hidden;
            }}
            #mapWrapper:fullscreen {{
                width: 100% !important;
                height: 100% !important;
                border-radius: 0 !important;
            }}
            #cesiumContainer {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            #searchPanel {{
                position: absolute;
                top: 12px;
                left: 50px;
                z-index: 1000;
                display: flex;
                gap: 6px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 6px 10px;
                width: 280px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }}
            #mapSearchInput {{
                flex: 1;
                background: transparent;
                border: none;
                outline: none;
                color: #ececf1;
                font-size: 0.8rem;
                font-family: inherit;
            }}
            #mapSearchBtn {{
                background: #00f0ff;
                color: #0f172a;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 0.75rem;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }}
            #mapSearchBtn:hover {{
                box-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
            }}
            #detailsPanel {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                right: 12px;
                z-index: 1000;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 0.82rem;
                color: #ececf1;
                line-height: 1.4;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
            }}
            .cesium-viewer-bottom {{ display: none !important; }}
        </style>
        <div id="mapWrapper">
            <div id="cesiumContainer"></div>
            <div id="searchPanel">
                 <input type="text" id="mapSearchInput" placeholder="Search any place or question...">
                 <button id="mapSearchBtn">Search</button>
            </div>
            <div id="detailsPanel">
                🌍 <b>Interactive 3D Globe</b><br/>
                Click any marker or any place on the sphere to fly to it and view historical/geographical details here.
            </div>
        </div>
        <script src="https://cesium.com/downloads/cesiumjs/releases/1.105/Build/Cesium/Cesium.js"></script>
        <script>
            Cesium.Ion.defaultAccessToken = '';

            try {{
                const viewer = new Cesium.Viewer('cesiumContainer', {{
                    imageryProvider: new Cesium.UrlTemplateImageryProvider({{
                        url: 'https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}',
                        maximumLevel: 20
                    }}),
                    fullscreenElement: document.getElementById('mapWrapper'),
                    fullscreenButton: true,
                    baseLayerPicker: false,
                    geocoder: false,
                    navigationHelpButton: false,
                    homeButton: false,
                    sceneModePicker: false,
                    timeline: false,
                    animation: false
                }});

                viewer.resize();
                setTimeout(() => {{ viewer.resize(); }}, 200);

                viewer.camera.setView({{
                    destination: Cesium.Cartesian3.fromDegrees(86.9250, 27.9881, 10000000.0)
                }});

                const pointsData = {json.dumps(points_data)};
                pointsData.forEach(p => {{
                    viewer.entities.add({{
                        position: Cesium.Cartesian3.fromDegrees(p.lng, p.lat),
                        billboard: {{
                            image: 'https://img.icons8.com/color/48/marker.png',
                            width: 32,
                            height: 32
                        }},
                        label: {{
                            text: p.name,
                            font: '14px Space Grotesk, sans-serif',
                            fillColor: Cesium.Color.AQUA,
                            outlineColor: Cesium.Color.BLACK,
                            outlineWidth: 2,
                            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                            pixelOffset: new Cesium.Cartesian2(0, -16)
                        }}
                    }});
                }});

                // selectedEntity handler for markers
                viewer.selectedEntityChanged.addEventListener(function(entity) {{
                    if (Cesium.defined(entity) && Cesium.defined(entity.label)) {{
                        const name = entity.label.text.getValue();
                        const descMap = {{
                            "Mount Everest": "Highest peak on Earth, located in the Himalayas. Famous for mountaineering and unique sub-zero alpine ecosystems.",
                            "Great Pyramids of Giza": "Ancient Egyptian pyramids near Cairo. Built during the Old Kingdom, they are marvels of ancient engineering.",
                            "Mariana Trench": "The deepest oceanic trench on Earth, situated in the western Pacific Ocean. Famous for extreme pressure and unique deep-sea life.",
                            "Amazon Rainforest": "World's largest tropical rainforest, stretching across South America. Home to unparalleled biodiversity and crucial global carbon sinks.",
                            "CERN (Hadron Collider)": "World's largest particle physics laboratory near Geneva. Famous for the Large Hadron Collider (LHC) and discovering the Higgs Boson."
                        }};
                        const desc = descMap[name] || "Famous academic and geographic site.";
                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + name + "</b><br/>" + desc;

                        const cartographic = Cesium.Ellipsoid.WGS84.cartesianToCartographic(entity.position.getValue(viewer.clock.currentTime));
                        const lat = Cesium.Math.toDegrees(cartographic.latitude);
                        const lng = Cesium.Math.toDegrees(cartographic.longitude);
                        window.parent.postMessage({{
                            type: 'globe_click',
                            name: name,
                            lat: lat,
                            lng: lng
                        }}, '*');
                    }}
                }});

                const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
                handler.setInputAction(function (click) {{
                    viewer.selectedEntity = undefined;
                    const pickedPosition = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
                    if (pickedPosition) {{
                        const cartographic = Cesium.Cartographic.fromCartesian(pickedPosition);
                        const lat = Cesium.Math.toDegrees(cartographic.latitude);
                        const lng = Cesium.Math.toDegrees(cartographic.longitude);
                        const coordStr = lat.toFixed(4) + ', ' + lng.toFixed(4);

                        document.getElementById('detailsPanel').innerHTML = "📍 <b>Coordinates: " + coordStr + "</b><br/>Click 'Analyze Location' in the sidebar to search detailed historical and geographical knowledge via Groq & Tavily AI.";

                        window.parent.postMessage({{
                            type: 'globe_click',
                            name: coordStr,
                            lat: lat,
                            lng: lng
                        }}, '*');
                    }}
                }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

                // Custom search input listeners
                const searchInput = document.getElementById('mapSearchInput');
                const searchBtn = document.getElementById('mapSearchBtn');

                function triggerSearch() {{
                    const query = searchInput.value.trim();
                    if (!query) return;

                    document.getElementById('detailsPanel').innerHTML = "🔍 <b>Searching: " + query + "...</b><br/>Fetching summary details and flying to target location...";

                    const wikiUrl = 'https://en.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(query);
                    fetch(wikiUrl)
                        .then(res => res.json())
                        .then(wikiData => {{
                            let desc = "No direct summary found. Click 'Analyze Location' in the sidebar to search via Groq AI.";
                            if (wikiData.extract) {{
                                desc = wikiData.extract;
                            }}

                            const geoUrl = 'https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(query) + '&limit=1';
                            fetch(geoUrl)
                                .then(r => r.json())
                                .then(geoData => {{
                                    if (geoData && geoData.length > 0) {{
                                        const lat = parseFloat(geoData[0].lat);
                                        const lon = parseFloat(geoData[0].lon);

                                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc;

                                        viewer.camera.flyTo({{
                                            destination: Cesium.Cartesian3.fromDegrees(lon, lat, 25000.0),
                                            duration: 2.0
                                        }});

                                        window.parent.postMessage({{
                                            type: 'globe_click',
                                            name: query,
                                            lat: lat,
                                            lng: lon
                                        }}, '*');
                                    }} else {{
                                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc + "<br/><span style='color:#ef4444;'>Failed to geocode location coordinates on map.</span>";
                                    }}
                                }})
                                .catch(err => {{
                                    document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc;
                                }});
                        }})
                        .catch(err => {{
                            document.getElementById('detailsPanel').innerHTML = "❌ <b>Error</b><br/>Failed to fetch search results.";
                        }});
                }}

                searchBtn.addEventListener('click', triggerSearch);
                searchInput.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        triggerSearch();
                    }}
                }});
            }} catch (e) {{
                console.error("Cesium failed to load", e);
                document.getElementById('cesiumContainer').innerHTML = "<div style='color:#ef4444; padding:20px; font-family:sans-serif;'>Failed to load 3D Globe: " + e.message + "</div>";
            }}
        </script>
        """
        st.components.v1.html(html_code, height=520)
        _card_close()

        st.markdown(
            """
            <script>
                if (!window.globeListenerAdded) {
                    window.addEventListener('message', function(event) {
                        if (event.data && event.data.type === 'globe_click') {
                            const name = event.data.name;
                            const nameInput = document.querySelector('input[aria-label="Location Name or Custom Coordinates"]');
                            if (nameInput) {
                                const nativeSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                                nativeSetter.call(nameInput, name);
                                nameInput.dispatchEvent(new Event('input', { bubbles: true }));
                                setTimeout(() => {
                                    const analyzeBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('Analyze Location'));
                                    if (analyzeBtn) {
                                        analyzeBtn.click();
                                    }
                                }, 100);
                            }
                        }
                    });
                    window.globeListenerAdded = true;
                }
            </script>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # ── Location Analyzer ──
        _card_open()
        st.markdown("#### 🔬 Location Analyzer")

        selected_place = st.text_input(
            "Location Name or Custom Coordinates",
            value="Mount Everest",
            help="Click a red label on the globe or type a custom place name/coordinates."
        )

        analyze_clicked = st.button("🔍 Analyze Location", use_container_width=True)
        _card_close()

        if analyze_clicked or selected_place:
            _card_open()
            st.markdown(f"##### 📚 Academic Report: **{selected_place}**")

            with logo_spinner(f"Retrieving research details for {selected_place}..."):
                search_query = f"{selected_place} geographical scientific historical facts summary"
                web_results = duckduckgo_search(search_query)
                web_context = ""
                if web_results:
                    web_context = "\n".join([res['snippet'] for res in web_results])

                system_role = (
                    "You are a helpful science, history, and geography AI tutor. "
                    "Write a neat, structured academic report on the requested location containing: "
                    "1. Coordinates & Geographic Overview "
                    "2. Historical & Cultural significance "
                    "3. Scientific / Geological / Environmental importance "
                    "4. Intriguing facts for students."
                )
                report = groq_chat(
                    f"Context: {web_context}\n\nProvide an academic report for: {selected_place}",
                    system=system_role
                )
                st.markdown(report)
            _card_close()



# ==============================================================================
# MODULE 14b — Standalone Weather Forecast (branch of 3D Globe)
# ==============================================================================

def render_weather_forecast() -> None:
    """⛅ Weather Forecast — live weather and 3-day forecast for any city, with AI insight."""
    st.markdown("### ⛅ Weather Forecast")
    st.caption("🌍 Branch of the Interactive 3D Globe · Powered by Open-Meteo (free, no API key required)")

    import urllib.request
    import urllib.parse
    import json as _json
    import datetime as _dt

    # ── WMO code → emoji + label ──
    def wx_icon(code: int) -> str:
        if code == 0:          return "☀️"
        if code in (1, 2, 3):  return "⛅"
        if code in (45, 48):   return "🌫️"
        if code in (51, 53, 55, 61, 63, 65): return "🌧️"
        if code in (71, 73, 75): return "❄️"
        if code in (80, 81, 82): return "🌦️"
        if code in (95, 96, 99): return "⛈️"
        return "🌡️"

    def wx_label(code: int) -> str:
        labels = {0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
                  45:"Fog", 48:"Icy fog", 51:"Light drizzle", 53:"Drizzle",
                  55:"Dense drizzle", 61:"Slight rain", 63:"Moderate rain", 65:"Heavy rain",
                  71:"Slight snow", 73:"Moderate snow", 75:"Heavy snow",
                  80:"Slight showers", 81:"Moderate showers", 82:"Violent showers",
                  95:"Thunderstorm", 96:"Thunderstorm w/ hail", 99:"Heavy thunderstorm"}
        return labels.get(code, "Unknown")

    # ── Search bar ──
    col_search, col_btn = st.columns([4, 1], gap="small")
    with col_search:
        weather_city = st.text_input(
            "🏙️ Enter City Name",
            value=st.session_state.get("wx_city", "Kathmandu"),
            key="wf_city_input",
            placeholder="e.g. London, Tokyo, New York, Kathmandu...",
            label_visibility="collapsed"
        )
    with col_btn:
        get_btn = st.button("🌡️ Get Weather", use_container_width=True, key="wf_get_btn")

    if not (get_btn and weather_city.strip()):
        st.markdown(
            """
            <div style="text-align:center; padding:40px; color:var(--sub); font-size:0.9rem;">
                🌍 Enter a city name above and click <b>Get Weather</b> to see live conditions
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    st.session_state["wx_city"] = weather_city.strip()

    with st.spinner(f"Fetching weather for {weather_city}..."):
        try:
            # Geocoding
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search"
                f"?name={urllib.parse.quote(weather_city)}&count=5&language=en&format=json"
            )
            with urllib.request.urlopen(geo_url, timeout=8) as resp:
                geo_data = _json.loads(resp.read())

            if not geo_data.get("results"):
                st.error(f"❌ City '{weather_city}' not found. Try a different spelling.")
                return

            r = geo_data["results"][0]
            lat_w, lon_w = r["latitude"], r["longitude"]
            country = r.get("country", "")
            admin  = r.get("admin1", "")
            city_name = r["name"]

            # Open-Meteo API
            wx_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat_w}&longitude={lon_w}"
                f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
                f"weather_code,apparent_temperature,precipitation,wind_direction_10m,"
                f"surface_pressure,visibility"
                f"&hourly=temperature_2m,weather_code"
                f"&daily=temperature_2m_max,temperature_2m_min,weather_code,"
                f"precipitation_sum,wind_speed_10m_max,sunrise,sunset,uv_index_max"
                f"&timezone=auto&forecast_days=7"
            )
            with urllib.request.urlopen(wx_url, timeout=10) as resp2:
                wx = _json.loads(resp2.read())

            cur   = wx["current"]
            daily = wx["daily"]

            temp   = cur["temperature_2m"]
            feels  = cur["apparent_temperature"]
            hum    = cur["relative_humidity_2m"]
            wind   = cur["wind_speed_10m"]
            wdir   = cur.get("wind_direction_10m", 0)
            precip = cur["precipitation"]
            press  = cur.get("surface_pressure", "—")
            vis    = cur.get("visibility", "—")
            code   = cur["weather_code"]
            icon   = wx_icon(code)
            label  = wx_label(code)

            # ── Main hero card ──
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,rgba(0,180,255,0.12),rgba(100,60,255,0.08));
                            border:1px solid rgba(0,200,255,0.2); border-radius:20px; padding:28px 32px;
                            margin-bottom:20px;">
                    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                        <div>
                            <div style="font-size:0.78rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
                                Live Conditions · {_dt.datetime.now().strftime('%H:%M, %d %b %Y')}
                            </div>
                            <div style="font-size:1.3rem; font-weight:700; color:var(--accent); margin-bottom:2px;">
                                📍 {city_name}, {admin}, {country}
                            </div>
                            <div style="font-size:0.8rem; color:var(--sub);">
                                {lat_w:.4f}°N · {lon_w:.4f}°E · {label}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:4rem; line-height:1;">{icon}</div>
                            <div style="font-size:2.6rem; font-weight:900; color:var(--text); font-family:'Orbitron',monospace;">{temp}°C</div>
                            <div style="font-size:0.85rem; color:var(--sub);">Feels like {feels}°C</div>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:22px;">
                        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px; text-align:center;">
                            <div style="font-size:1.3rem;">💧</div>
                            <div style="font-size:1rem; font-weight:700; color:var(--text);">{hum}%</div>
                            <div style="font-size:0.68rem; color:var(--sub);">Humidity</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px; text-align:center;">
                            <div style="font-size:1.3rem;">💨</div>
                            <div style="font-size:1rem; font-weight:700; color:var(--text);">{wind} km/h</div>
                            <div style="font-size:0.68rem; color:var(--sub);">Wind ({wdir}°)</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px; text-align:center;">
                            <div style="font-size:1.3rem;">🌧️</div>
                            <div style="font-size:1rem; font-weight:700; color:var(--text);">{precip} mm</div>
                            <div style="font-size:0.68rem; color:var(--sub);">Precipitation</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:12px; text-align:center;">
                            <div style="font-size:1.3rem;">🔵</div>
                            <div style="font-size:1rem; font-weight:700; color:var(--text);">{press} hPa</div>
                            <div style="font-size:0.68rem; color:var(--sub);">Pressure</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── 7-Day Forecast ──
            st.markdown("#### 📅 7-Day Forecast")
            days_count = min(7, len(daily["time"]))
            day_cols = st.columns(days_count)
            for i in range(days_count):
                day_label = _dt.date.fromisoformat(daily["time"][i]).strftime("%a\n%d %b")
                dc = daily["weather_code"][i]
                sunrise = daily.get("sunrise", ["—"])[i].split("T")[-1][:5] if "sunrise" in daily else "—"
                sunset  = daily.get("sunset", ["—"])[i].split("T")[-1][:5] if "sunset" in daily else "—"
                uv = daily.get("uv_index_max", [0])[i] if "uv_index_max" in daily else 0
                with day_cols[i]:
                    is_today = i == 0
                    border = "border:1px solid rgba(0,240,255,0.35);" if is_today else "border:1px solid rgba(255,255,255,0.07);"
                    st.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,{'0.08' if is_today else '0.03'});
                                    {border} border-radius:14px; padding:12px 8px; text-align:center;">
                            <div style="font-size:0.68rem; color:{'var(--accent)' if is_today else 'var(--sub)'}; font-weight:{'700' if is_today else '400'}; white-space:pre-line; margin-bottom:6px;">
                                {day_label}{'\\n(Today)' if is_today else ''}
                            </div>
                            <div style="font-size:1.8rem; margin:4px 0;">{wx_icon(dc)}</div>
                            <div style="color:var(--accent); font-weight:700; font-size:0.88rem;">{daily['temperature_2m_max'][i]}°</div>
                            <div style="color:var(--sub); font-size:0.78rem;">{daily['temperature_2m_min'][i]}°</div>
                            <div style="font-size:0.65rem; color:var(--sub); margin-top:6px; line-height:1.6;">
                                💧{daily['precipitation_sum'][i]}mm<br>
                                ☀{sunrise}<br>
                                🌙{sunset}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ── Hourly temperature chart (simple HTML sparkline) ──
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📈 Hourly Temperature (Next 24h)")
            hourly_temps = wx["hourly"]["temperature_2m"][:24]
            hourly_times = [t.split("T")[1][:5] for t in wx["hourly"]["time"][:24]]
            chart_data = {"Time": hourly_times, "Temp (°C)": hourly_temps}
            try:
                import pandas as pd
                df = pd.DataFrame(chart_data).set_index("Time")
                st.line_chart(df, use_container_width=True, height=180)
            except Exception:
                st.caption("Chart unavailable (pandas not installed).")

            # ── AI Weather Commentary ──
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            _card_open()
            st.markdown("#### 🤖 AI Weather Insight")
            with logo_spinner("Generating AI weather report..."):
                wx_prompt = (
                    f"Current weather in {city_name}, {country}: {temp}°C (feels {feels}°C), "
                    f"humidity {hum}%, wind {wind} km/h ({wdir}°), precipitation {precip}mm, "
                    f"pressure {press} hPa. Condition: {label}. "
                    f"7-day max temps: {daily['temperature_2m_max'][:7]}. "
                    f"Write a friendly 3-sentence weather summary, mention any notable conditions, "
                    f"and give 2 practical tips for students going outside today."
                )
                insight = groq_chat(wx_prompt, system="You are a helpful and friendly meteorology assistant.")
                st.markdown(insight)
            _card_close()

            # ── Other cities (if multiple results) ──
            if len(geo_data["results"]) > 1:
                st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                with st.expander("📍 Other matching cities"):
                    for alt in geo_data["results"][1:]:
                        st.markdown(
                            f"**{alt['name']}** — {alt.get('admin1','')}, {alt.get('country','')} "
                            f"({alt['latitude']:.3f}°, {alt['longitude']:.3f}°)"
                        )

        except Exception as err:
            st.error(f"❌ Weather fetch failed: {err}")





        get_weather = st.button("🌡️ Get Weather", use_container_width=True, key="get_weather_btn")

        if get_weather and weather_city.strip():
            with st.spinner(f"Fetching weather for {weather_city}..."):
                try:
                    import urllib.request
                    import json as _json

                    # Open-Meteo geocoding (free, no key)
                    geo_url = (
                        f"https://geocoding-api.open-meteo.com/v1/search"
                        f"?name={urllib.parse.quote(weather_city)}&count=1&language=en&format=json"
                    )
                    import urllib.parse
                    with urllib.request.urlopen(geo_url, timeout=8) as resp:
                        geo_data = _json.loads(resp.read())

                    if not geo_data.get("results"):
                        st.error(f"❌ City '{weather_city}' not found.")
                    else:
                        r = geo_data["results"][0]
                        lat_w, lon_w = r["latitude"], r["longitude"]
                        country = r.get("country", "")
                        admin = r.get("admin1", "")

                        # Open-Meteo weather API (free, no key)
                        wx_url = (
                            f"https://api.open-meteo.com/v1/forecast"
                            f"?latitude={lat_w}&longitude={lon_w}"
                            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
                            f"weather_code,apparent_temperature,precipitation"
                            f"&daily=temperature_2m_max,temperature_2m_min,weather_code,"
                            f"precipitation_sum,wind_speed_10m_max"
                            f"&timezone=auto&forecast_days=4"
                        )
                        with urllib.request.urlopen(wx_url, timeout=8) as resp2:
                            wx = _json.loads(resp2.read())

                        cur = wx["current"]
                        daily = wx["daily"]

                        # WMO weather code → emoji
                        def wx_icon(code):
                            if code == 0: return "☀️"
                            if code in (1, 2, 3): return "⛅"
                            if code in (45, 48): return "🌫️"
                            if code in (51, 53, 55, 61, 63, 65): return "🌧️"
                            if code in (71, 73, 75): return "❄️"
                            if code in (80, 81, 82): return "🌦️"
                            if code in (95, 96, 99): return "⛈️"
                            return "🌡️"

                        temp = cur["temperature_2m"]
                        feels = cur["apparent_temperature"]
                        hum = cur["relative_humidity_2m"]
                        wind = cur["wind_speed_10m"]
                        precip = cur["precipitation"]
                        icon = wx_icon(cur["weather_code"])

                        st.markdown(
                            f"""
                            <div style="background:rgba(0,240,255,0.06); border:1px solid rgba(0,240,255,0.18);
                                        border-radius:14px; padding:16px 18px; margin-bottom:12px;">
                                <div style="font-size:2.6rem; text-align:center; margin-bottom:4px;">{icon}</div>
                                <div style="text-align:center; font-size:1rem; font-weight:700;
                                            color:var(--accent); margin-bottom:2px;">
                                    {r['name']}, {admin}, {country}
                                </div>
                                <div style="text-align:center; font-size:2rem; font-weight:900;
                                            color:var(--text); margin-bottom:8px;">{temp}°C</div>
                                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.78rem; color:var(--sub);">
                                    <div>🌡️ Feels like: <b style="color:var(--text);">{feels}°C</b></div>
                                    <div>💧 Humidity: <b style="color:var(--text);">{hum}%</b></div>
                                    <div>💨 Wind: <b style="color:var(--text);">{wind} km/h</b></div>
                                    <div>🌧️ Precip: <b style="color:var(--text);">{precip} mm</b></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # 3-day forecast
                        st.markdown("**📅 3-Day Forecast**")
                        import datetime as _dt
                        days = daily["time"][1:4]
                        max_t = daily["temperature_2m_max"][1:4]
                        min_t = daily["temperature_2m_min"][1:4]
                        codes = daily["weather_code"][1:4]
                        precips = daily["precipitation_sum"][1:4]
                        winds = daily["wind_speed_10m_max"][1:4]

                        fc_html = '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:8px;">'
                        for i in range(3):
                            day_label = _dt.date.fromisoformat(days[i]).strftime("%a %d %b")
                            fc_html += f"""
                            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                        border-radius:10px; padding:10px 8px; text-align:center; font-size:0.78rem;">
                                <div style="font-size:0.7rem; color:var(--sub); margin-bottom:4px;">{day_label}</div>
                                <div style="font-size:1.6rem;">{wx_icon(codes[i])}</div>
                                <div style="color:var(--accent); font-weight:700;">{max_t[i]}°C</div>
                                <div style="color:var(--sub);">{min_t[i]}°C</div>
                                <div style="color:var(--sub); font-size:0.68rem; margin-top:4px;">
                                    💧{precips[i]}mm 💨{winds[i]}km/h
                                </div>
                            </div>
                            """
                        fc_html += "</div>"
                        st.markdown(fc_html, unsafe_allow_html=True)

                        # AI weather commentary
                        with logo_spinner("Generating AI weather insight..."):
                            wx_prompt = (
                                f"The current weather in {r['name']}, {country} is {temp}°C (feels like {feels}°C), "
                                f"humidity {hum}%, wind {wind} km/h, precipitation {precip}mm. "
                                f"Give a 2-sentence friendly weather summary and one practical tip for students."
                            )
                            wx_insight = groq_chat(wx_prompt, system="You are a helpful meteorology assistant.")
                            st.info(f"🤖 **AI Insight:** {wx_insight}")

                except Exception as wx_err:
                    st.error(f"❌ Weather fetch failed: {wx_err}")

        _card_close()


        _card_open()
        st.markdown("#### 🗺️ 3D Globe Viewer (Drag to rotate, scroll to zoom)")
        
        import json
        points_data = [
            {"lat": val["lat"], "lng": val["lng"], "name": name, "color": "red", "size": 0.5}
            for name, val in PREDEFINED_PLACES.items()
        ]
        
        html_code = f"""
        <link href="https://cesium.com/downloads/cesiumjs/releases/1.105/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                background: #000;
                font-family: 'Space Grotesk', sans-serif;
            }}
            #mapWrapper {{
                position: relative;
                width: 100%;
                height: 500px;
                border-radius: 8px;
                overflow: hidden;
            }}
            #mapWrapper:fullscreen {{
                width: 100% !important;
                height: 100% !important;
                border-radius: 0 !important;
            }}
            #cesiumContainer {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            #searchPanel {{
                position: absolute;
                top: 12px;
                left: 50px;
                z-index: 1000;
                display: flex;
                gap: 6px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 6px 10px;
                width: 280px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }}
            #mapSearchInput {{
                flex: 1;
                background: transparent;
                border: none;
                outline: none;
                color: #ececf1;
                font-size: 0.8rem;
                font-family: inherit;
            }}
            #mapSearchBtn {{
                background: #00f0ff;
                color: #0f172a;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 0.75rem;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }}
            #mapSearchBtn:hover {{
                box-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
            }}
            #detailsPanel {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                right: 12px;
                z-index: 1000;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 0.82rem;
                color: #ececf1;
                line-height: 1.4;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
            }}
            .cesium-viewer-bottom {{ display: none !important; }}
        </style>
        <div id="mapWrapper">
            <div id="cesiumContainer"></div>
            <div id="searchPanel">
                 <input type="text" id="mapSearchInput" placeholder="Search any place or question...">
                 <button id="mapSearchBtn">Search</button>
            </div>
            <div id="detailsPanel">
                🌍 <b>Interactive 3D Globe</b><br/>
                Click any marker or any place on the sphere to fly to it and view historical/geographical details here.
            </div>
        </div>
        <script src="https://cesium.com/downloads/cesiumjs/releases/1.105/Build/Cesium/Cesium.js"></script>
        <script>
            Cesium.Ion.defaultAccessToken = '';
            
            try {{
                const viewer = new Cesium.Viewer('cesiumContainer', {{
                    imageryProvider: new Cesium.UrlTemplateImageryProvider({{
                        url: 'https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}',
                        maximumLevel: 20
                    }}),
                    fullscreenElement: document.getElementById('mapWrapper'),
                    fullscreenButton: true,
                    baseLayerPicker: false,
                    geocoder: false,
                    navigationHelpButton: false,
                    homeButton: false,
                    sceneModePicker: false,
                    timeline: false,
                    animation: false
                }});
                
                viewer.resize();
                setTimeout(() => {{ viewer.resize(); }}, 200);

                viewer.camera.setView({{
                    destination: Cesium.Cartesian3.fromDegrees(86.9250, 27.9881, 10000000.0)
                }});

                const pointsData = {json.dumps(points_data)};
                pointsData.forEach(p => {{
                    viewer.entities.add({{
                        position: Cesium.Cartesian3.fromDegrees(p.lng, p.lat),
                        billboard: {{
                            image: 'https://img.icons8.com/color/48/marker.png',
                            width: 32,
                            height: 32
                        }},
                        label: {{
                            text: p.name,
                            font: '14px Space Grotesk, sans-serif',
                            fillColor: Cesium.Color.AQUA,
                            outlineColor: Cesium.Color.BLACK,
                            outlineWidth: 2,
                            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                            pixelOffset: new Cesium.Cartesian2(0, -16)
                        }}
                    }});
                }});

                // selectedEntity handler for markers
                viewer.selectedEntityChanged.addEventListener(function(entity) {{
                    if (Cesium.defined(entity) && Cesium.defined(entity.label)) {{
                        const name = entity.label.text.getValue();
                        const descMap = {{
                            "Mount Everest": "Highest peak on Earth, located in the Himalayas. Famous for mountaineering and unique sub-zero alpine ecosystems.",
                            "Great Pyramids of Giza": "Ancient Egyptian pyramids near Cairo. Built during the Old Kingdom, they are marvels of ancient engineering.",
                            "Mariana Trench": "The deepest oceanic trench on Earth, situated in the western Pacific Ocean. Famous for extreme pressure and unique deep-sea life.",
                            "Amazon Rainforest": "World's largest tropical rainforest, stretching across South America. Home to unparalleled biodiversity and crucial global carbon sinks.",
                            "CERN (Hadron Collider)": "World's largest particle physics laboratory near Geneva. Famous for the Large Hadron Collider (LHC) and discovering the Higgs Boson."
                        }};
                        const desc = descMap[name] || "Famous academic and geographic site.";
                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + name + "</b><br/>" + desc;
                        
                        const cartographic = Cesium.Ellipsoid.WGS84.cartesianToCartographic(entity.position.getValue(viewer.clock.currentTime));
                        const lat = Cesium.Math.toDegrees(cartographic.latitude);
                        const lng = Cesium.Math.toDegrees(cartographic.longitude);
                        window.parent.postMessage({{
                            type: 'globe_click',
                            name: name,
                            lat: lat,
                            lng: lng
                        }}, '*');
                    }}
                }});

                const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
                handler.setInputAction(function (click) {{
                    viewer.selectedEntity = undefined;
                    const pickedPosition = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
                    if (pickedPosition) {{
                        const cartographic = Cesium.Cartographic.fromCartesian(pickedPosition);
                        const lat = Cesium.Math.toDegrees(cartographic.latitude);
                        const lng = Cesium.Math.toDegrees(cartographic.longitude);
                        const coordStr = lat.toFixed(4) + ', ' + lng.toFixed(4);
                        
                        document.getElementById('detailsPanel').innerHTML = "📍 <b>Coordinates: " + coordStr + "</b><br/>Click 'Analyze Location' in the sidebar to search detailed historical and geographical knowledge via Groq & Tavily AI.";
                        
                        window.parent.postMessage({{
                            type: 'globe_click',
                            name: coordStr,
                            lat: lat,
                            lng: lng
                        }}, '*');
                    }}
                }}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

                // Custom search input listeners
                const searchInput = document.getElementById('mapSearchInput');
                const searchBtn = document.getElementById('mapSearchBtn');

                function triggerSearch() {{
                    const query = searchInput.value.trim();
                    if (!query) return;
                    
                    document.getElementById('detailsPanel').innerHTML = "🔍 <b>Searching: " + query + "...</b><br/>Fetching summary details and flying to target location...";
                    
                    const wikiUrl = 'https://en.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(query);
                    fetch(wikiUrl)
                        .then(res => res.json())
                        .then(wikiData => {{
                            let desc = "No direct summary found. Click 'Analyze Location' in the sidebar to search via Groq AI.";
                            if (wikiData.extract) {{
                                desc = wikiData.extract;
                            }}
                            
                            const geoUrl = 'https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(query) + '&limit=1';
                            fetch(geoUrl)
                                .then(r => r.json())
                                .then(geoData => {{
                                    if (geoData && geoData.length > 0) {{
                                        const lat = parseFloat(geoData[0].lat);
                                        const lon = parseFloat(geoData[0].lon);
                                        
                                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc;
                                        
                                        viewer.camera.flyTo({{
                                            destination: Cesium.Cartesian3.fromDegrees(lon, lat, 25000.0),
                                            duration: 2.0
                                        }});
                                        
                                        window.parent.postMessage({{
                                            type: 'globe_click',
                                            name: query,
                                            lat: lat,
                                            lng: lon
                                        }}, '*');
                                    }} else {{
                                        document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc + "<br/><span style='color:#ef4444;'>Failed to geocode location coordinates on map.</span>";
                                    }}
                                }})
                                .catch(err => {{
                                    document.getElementById('detailsPanel').innerHTML = "📍 <b>" + (wikiData.title || query) + "</b><br/>" + desc;
                                }});
                        }})
                        .catch(err => {{
                            document.getElementById('detailsPanel').innerHTML = "❌ <b>Error</b><br/>Failed to fetch search results.";
                        }});
                }}

                searchBtn.addEventListener('click', triggerSearch);
                searchInput.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        triggerSearch();
                    }}
                }});
            }} catch (e) {{
                console.error("Cesium failed to load", e);
                document.getElementById('cesiumContainer').innerHTML = "<div style='color:#ef4444; padding:20px; font-family:sans-serif;'>Failed to load 3D Globe: " + e.message + "</div>";
            }}
        </script>
        """
        st.components.v1.html(html_code, height=520)
        _card_close()
        
        st.markdown(
            """
            <script>
                if (!window.globeListenerAdded) {
                    window.addEventListener('message', function(event) {
                        if (event.data && event.data.type === 'globe_click') {
                            const name = event.data.name;
                            const nameInput = document.querySelector('input[aria-label="Location Name or Custom Coordinates"]');
                            if (nameInput) {
                                const nativeSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                                nativeSetter.call(nameInput, name);
                                nameInput.dispatchEvent(new Event('input', { bubbles: true }));
                                setTimeout(() => {
                                    const analyzeBtn = Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes('Analyze Location'));
                                    if (analyzeBtn) {
                                        analyzeBtn.click();
                                    }
                                }, 100);
                            }
                        }
                    });
                    window.globeListenerAdded = true;
                }
            </script>
            """,
            unsafe_allow_html=True
        )

    with col2:
        _card_open()
        st.markdown("#### 🔬 Location Analyzer")
        
        selected_place = st.text_input(
            "Location Name or Custom Coordinates",
            value="Mount Everest",
            help="Click a red label on the globe or type a custom place name/coordinates."
        )
        
        analyze_clicked = st.button("🔍 Analyze Location", use_container_width=True)
        _card_close()
        
        if analyze_clicked or selected_place:
            _card_open()
            st.markdown(f"##### 📚 Academic Report: **{selected_place}**")
            
            with logo_spinner(f"Retrieving research details for {selected_place}..."):
                search_query = f"{selected_place} geographical scientific historical facts summary"
                web_results = duckduckgo_search(search_query)
                web_context = ""
                if web_results:
                    web_context = "\n".join([res['snippet'] for res in web_results])
                
                system_role = (
                    "You are a helpful science, history, and geography AI tutor. "
                    "Write a neat, structured academic report on the requested location containing: "
                    "1. Coordinates & Geographic Overview "
                    "2. Historical & Cultural significance "
                    "3. Scientific / Geological / Environmental importance "
                    "4. Intriguing facts for students."
                )
                report = groq_chat(
                    f"Context: {web_context}\n\nProvide an academic report for: {selected_place}",
                    system=system_role
                )
                st.markdown(report)
            _card_close()


# ==============================================================================
# MODULE 15 — Cyber Security Panel
# ==============================================================================

def render_cyber_panel() -> None:
    """🛡️ Cyber Security Panel — Analyze Spam, Phishing, Malware headers, and Threat Feeds."""
    st.markdown("### 🛡️ Cyber Security Threat & File Analysis Panel")
    
    tab_email, tab_malware, tab_feeds = st.tabs([
        "📧 Email & Link Safety",
        "🪱 Malware & File Diagnostics",
        "📡 Live Threat Intelligence"
    ])
    
    with tab_email:
        _card_open()
        st.markdown("#### 📧 Email Header / Text / Link Reputation Analyzer")
        
        email_content = st.text_area(
            "Paste Email Content (Subject, Body, Headers, or Links)",
            placeholder="e.g. Dear customer, your account is suspended. Click here http://bank-verify-security.com to login...",
            height=150,
            key="cyber_email_text"
        )
        
        analyze_email = st.button("🛡️ Analyze Threat Vector", key="btn_analyze_email")
        _card_close()
        
        if analyze_email:
            if not email_content.strip():
                st.warning("Please paste some email content or links to analyze.")
            else:
                _card_open()
                st.markdown("##### 🔍 Cyber Analyst Report")
                
                with logo_spinner("Running threat model analysis..."):
                    import re
                    links = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', email_content)
                    link_context = ""
                    if links:
                        link_context = "Found Links:\n" + "\n".join(links) + "\n\n"
                    
                    system_role = (
                        "You are an expert Cyber Security Analyst and Incident Responder. "
                        "Inspect the provided email/link threat vector. Detect signs of: "
                        "- Phishing indicators (urgent tone, generic greetings, spelling errors, mismatched domains) "
                        "- Spam markers "
                        "- Link reputation (check if domains look spoofed or suspicious) "
                        "Assign a Threat Level (Low / Medium / High / Critical) and outline mitigations."
                    )
                    
                    report = groq_chat(
                        f"{link_context}Email/Link Content to analyze:\n{email_content}",
                        system=system_role
                    )
                    st.markdown(report)
                _card_close()
                
    with tab_malware:
        _card_open()
        st.markdown("#### 🪱 File Structure & Malware Header Integrity Diagnostics")
        st.info("Upload any file to analyze its headers (Magic Bytes) for integrity and potential masquerading.")
        
        uploaded_file = st.file_uploader(
            "Upload File to Scan",
            key="cyber_file_uploader",
            help="Select any file (PNG, PDF, EXE, TXT, etc.). We inspect its header signature safely."
        )
        _card_close()
        
        if uploaded_file:
            _card_open()
            file_bytes = uploaded_file.getvalue()
            file_size = len(file_bytes)
            
            header_hex = file_bytes[:16].hex().upper()
            header_space = " ".join([header_hex[i:i+2] for i in range(0, len(header_hex), 2)])
            
            signatures = {
                "89 50 4E 47 0D 0A 1A 0A": "PNG Image File",
                "25 50 44 46": "PDF Document",
                "FF D8 FF": "JPEG Image File",
                "4D 5A": "Windows Executable (EXE / DLL)",
                "50 4B 03 04": "ZIP / Office OpenXML Archive (DOCX/PPTX/ZIP)",
                "7F 45 4C 46": "ELF Executable (Linux/Unix)",
                "49 44 33": "MP3 Audio File",
                "52 61 72 21 1A 07": "RAR Compressed Archive"
            }
            
            detected_format = "Unknown / Custom Binary Data"
            for sig, label in signatures.items():
                sig_clean = sig.replace(" ", "")
                if header_hex.startswith(sig_clean):
                    detected_format = label
                    break
                    
            st.markdown(f"**File Name:** `{uploaded_file.name}`")
            st.markdown(f"**File Size:** `{file_size} bytes`")
            st.markdown(f"**Magic Bytes (Hex Header):** `{header_space}`")
            st.markdown(f"**Detected Signature Format:** `{detected_format}`")
            
            with logo_spinner("Evaluating file header integrity and safety profile..."):
                prompt = (
                    f"File Name: {uploaded_file.name}\n"
                    f"File Size: {file_size} bytes\n"
                    f"Header Hex: {header_hex}\n"
                    f"Signature Match: {detected_format}\n\n"
                    "Analyze if the file extension matches the signature (masquerading check), "
                    "explain what this file signature means, and assess security concerns if any."
                )
                analysis = groq_chat(
                    prompt,
                    system="You are a Malware Analysis specialist. Provide a brief, professional integrity report."
                )
                st.markdown("##### 🧬 File Integrity Report")
                st.markdown(analysis)
            _card_close()
            
    with tab_feeds:
        _card_open()
        st.markdown("#### 📡 Global Cyber Security Threats & Vulnerability Feed")
        st.info("Fetches real-time security alerts, CVE releases, and cyber attack reports using Tavily Search.")
        
        search_keyword = st.text_input("Vulnerability / Threat Keyword Search", value="Recent ransomware campaigns zero-day exploits")
        fetch_clicked = st.button("📡 Fetch Threat Intel Feed", use_container_width=True)
        _card_close()
        
        if fetch_clicked or search_keyword:
            _card_open()
            st.markdown(f"##### 📢 Intel Report for: *{search_keyword}*")
            
            with logo_spinner(f"Scanning web for threat intel on '{search_keyword}'..."):
                results = duckduckgo_search(f"{search_keyword} security threat alert news cve")
                if not results:
                    st.warning("No recent threat reports found.")
                else:
                    context_pieces = []
                    for idx, res in enumerate(results[:5], 1):
                        context_pieces.append(
                            f"**[{idx}] {res['title']}**\n"
                            f"Source: {res['link']}\n"
                            f"Intel Summary: {res['snippet']}\n"
                        )
                        st.markdown(f"🔗 [{res['title']}]({res['link']})")
                        st.write(res['snippet'])
                        st.markdown("---")
                        
                    intel_context = "\n".join(context_pieces)
                    report = groq_chat(
                        f"Intel Context:\n{intel_context}\n\nProvide a concise executive threat brief summarizing the main risks, active CVEs, and recommended protection steps.",
                        system="You are a Cyber Threat Intelligence Specialist. Provide a clear, actionable brief."
                    )
                    st.markdown("##### 📝 Threat Intelligence Executive Brief")
                    st.markdown(report)
            _card_close()
