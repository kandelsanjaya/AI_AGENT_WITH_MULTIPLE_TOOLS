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
# MODULE 1 — RAG Chatbot (with LICT Knowledge Integration)
# ==============================================================================

def render_educhat() -> None:
    """🧠 EduChat & RAG Studio — document-aware conversational AI with LICT campus knowledge."""
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
        # Scrollable chat history area
        chat_container = st.container(height=420)
        with chat_container:
            _render_chat_history()

        # ── Slim voice-only strip above the native input ─────────────────────
        st.components.v1.html(
            """
            <div style="font-family:'Inter',sans-serif; display:flex; align-items:center; gap:8px;
                        background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px; padding:5px 14px; margin-bottom:4px;">
                <button type="button" id="mic-btn"
                    style="background:transparent;border:none;font-size:1.2rem;color:#94a3b8;
                           cursor:pointer;outline:none;padding:0;transition:color .2s;">🎙️</button>
                <select id="lang-select"
                    style="background:#1e1e24;color:#94a3b8;border:1px solid rgba(255,255,255,0.15);
                           font-size:0.75rem;outline:none;border-radius:6px;padding:2px 5px;">
                    <option value="en-US">EN</option>
                    <option value="ne-NP">NE</option>
                    <option value="hi-IN">HI</option>
                    <option value="es-ES">ES</option>
                    <option value="fr-FR">FR</option>
                    <option value="zh-CN">ZH</option>
                </select>
                <span id="status" style="color:#64748b;font-size:0.78rem;font-style:italic;">
                    Click 🎙️ to speak — transcribed text will appear in the box below
                </span>
            </div>
            <script>
                const micBtn = document.getElementById('mic-btn');
                const langSel = document.getElementById('lang-select');
                const statusEl = document.getElementById('status');

                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SR) {
                    micBtn.style.display = 'none';
                    statusEl.textContent = 'Voice not supported in this browser.';
                } else {
                    const rec = new SR();
                    rec.continuous = false;
                    rec.interimResults = false;

                    micBtn.addEventListener('click', () => {
                        rec.lang = langSel.value;
                        rec.start();
                        micBtn.style.color = '#ef4444';
                        statusEl.textContent = '🔴 Listening…';
                    });

                    rec.onresult = (e) => {
                        const text = e.results[0][0].transcript;
                        statusEl.textContent = '✅ Transcribed: ' + text + ' — sending…';
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set('voice_prompt', text);
                        window.parent.location.href = url.toString();
                    };

                    rec.onspeechend = () => rec.stop();
                    rec.onerror = (e) => {
                        micBtn.style.color = '#94a3b8';
                        statusEl.textContent = '⚠️ Error: ' + e.error;
                    };
                    rec.onend = () => {
                        micBtn.style.color = '#94a3b8';
                    };
                }
            </script>
            """,
            height=52,
        )

        # ── Native Streamlit chat input ──
        prompt = st.chat_input("💬 Ask EduSphere anything…", key="edu_chat_input")

        # Also check for voice query param (voice path)
        voice_param = st.query_params.get("voice_prompt", "")
        if voice_param and not prompt:
            prompt = voice_param
            st.query_params.clear()

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
            st.rerun()

        # Guardrails
        guards = evaluate_guardrails(prompt)
        if guards.crisis:
            st.session_state.blocked_count = st.session_state.get("blocked_count", 0) + 1
            st.markdown(
                f'<div class="g-card" style="border-color:#ef4444;">{CRISIS_RESOURCES}</div>',
                unsafe_allow_html=True,
            )
            return
        if guards.harmful or guards.private:
            st.session_state.blocked_count = st.session_state.get("blocked_count", 0) + 1
            st.error("🚫 Prompt blocked: Violates System Security or Data Privacy Policy.")
            return

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

        if not context:
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

        if used_web_search:
            system_role = (
                "You are an expert, encouraging academic AI tutor and College AI Assistant. "
                "Answer questions precisely based on the Live Web Search Results context provided. "
                "Cite your web sources (e.g. Source [1], Source [2], etc.) and links in the response. "
                "CRITICAL: You must write your entire explanation and answer in the same language as the user's question."
            )
        else:
            system_role = (
                "You are an expert, encouraging academic AI tutor and College AI Assistant. "
                "Answer questions precisely based on context provided if available. "
                "If college knowledge context is provided, ground your answer in it and mention it naturally. "
                "If context is provided, cite relevant sections. "
                "If not, answer from your training knowledge. "
                "CRITICAL: You must write your entire explanation and answer in the same language as the user's question."
            )

        # Collect response and store, then rerun to show inside the scrollable container
        with logo_spinner("🤖 EduSphere is thinking…"):
            response_text = groq_chat(full_prompt, system=system_role)

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

    input_text = st.text_area(
        "Source Text",
        "Neural networks utilise backpropagation to update weights based on gradient loss.",
        height=120,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        target_lang = st.selectbox(
            "Target Language",
            ["Nepali", "Spanish", "French", "German", "Chinese (Mandarin)", "Japanese", "Hindi", "Arabic"],
        )
    with col2:
        formality = st.selectbox("Formality Level", ["Academic / Formal", "Conversational", "Technical"])

    if st.button("🌐 Translate Text", key="btn_translate"):
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

        st.markdown("---")
        st.markdown("#### ⬇️ Download Your Resume")
        _render_export_buttons(resume_text, f"resume_{full_name.replace(' ', '_')}")
