def render_dashboard() -> None:
    """≡ƒÅá EduSphere AI Dashboard ΓÇö Lucy-AI style workspace home."""
    user = st.session_state.user_info or {}
    name = user.get("name", "User")
    role = user.get("role", "Student")
    now = datetime.datetime.now()
    hour = now.hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    total_q = st.session_state.get("total_queries", 0)
    chat_len = len(st.session_state.get("chat_history", []))
    session_mins = int((now - st.session_state.get("session_start", now)).total_seconds() / 60)

    # ΓöÇΓöÇ Stat Row ΓöÇΓöÇ
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
                    <div style="font-size:0.72rem; color:#4ade80; margin-top:6px;">Γû▓ Active session</div>
                </div>
                <div style="background:var(--card); border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px; padding:18px 20px;">
                    <div style="font-size:0.68rem; color:var(--sub); letter-spacing:1px; text-transform:uppercase;
                                margin-bottom:8px;">Chat Messages</div>
                    <div style="font-size:2rem; font-weight:800; color:var(--text); font-family:'Orbitron',monospace;">{chat_len}</div>
                    <div style="font-size:0.72rem; color:var(--accent); margin-top:6px;">Γû▓ This session</div>
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
                    <div style="font-size:0.72rem; color:#fb923c; margin-top:6px;">ΓùÅ Authenticated</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ΓöÇΓöÇ Main area + right panel ΓöÇΓöÇ
    left_col, right_col = st.columns([2.2, 1], gap="medium")

    with left_col:
        # Greeting card ΓÇö UIverse lava-lamp orb (rendered inside streamlit component iframe to prevent sanitization)
        _card_open()

        import streamlit.components.v1 as _stc_orb

        _greeting_v = greeting
        _name_v     = name

        _html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: transparent;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    overflow: hidden;
  }

  .loader {
    --color-one: #ffbf48;
    --color-two: #be4a1d;
    --color-three: #ffbf4780;
    --color-four: #bf4a1d80;
    --color-five: #ffbf4740;
    --time-animation: 2s;
    --size: 1; /* You can change the size */
    position: relative;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    transform: scale(var(--size));
    box-shadow:
      0 0 25px 0 var(--color-three),
      0 20px 50px 0 var(--color-four);
    animation: colorize calc(var(--time-animation) * 3) ease-in-out infinite;
  }

  .loader::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    border-top: solid 1px var(--color-one);
    border-bottom: solid 1px var(--color-two);
    background: linear-gradient(180deg, var(--color-five), var(--color-four));
    box-shadow:
      inset 0 10px 10px 0 var(--color-three),
      inset 0 -10px 10px 0 var(--color-four);
  }

  .loader .box {
    width: 100px;
    height: 100px;
    background: linear-gradient(
      180deg,
      var(--color-one) 30%,
      var(--color-two) 70%
    );
    mask: url(#clipping);
    -webkit-mask: url(#clipping);
  }

  .loader svg {
    position: absolute;
  }

  .loader svg #clipping {
    filter: contrast(15);
    animation: roundness calc(var(--time-animation) / 2) linear infinite;
  }

  .loader svg #clipping polygon {
    filter: blur(7px);
  }

  .loader svg #clipping polygon:nth-child(1) {
    transform-origin: 75% 25%;
    transform: rotate(90deg);
  }

  .loader svg #clipping polygon:nth-child(2) {
    transform-origin: 50% 50%;
    animation: rotation var(--time-animation) linear infinite reverse;
  }

  .loader svg #clipping polygon:nth-child(3) {
    transform-origin: 50% 60%;
    animation: rotation var(--time-animation) linear infinite;
    animation-delay: calc(var(--time-animation) / -3);
  }

  .loader svg #clipping polygon:nth-child(4) {
    transform-origin: 40% 40%;
    animation: rotation var(--time-animation) linear infinite reverse;
  }

  .loader svg #clipping polygon:nth-child(5) {
    transform-origin: 40% 40%;
    animation: rotation var(--time-animation) linear infinite reverse;
    animation-delay: calc(var(--time-animation) / -2);
  }

  .loader svg #clipping polygon:nth-child(6) {
    transform-origin: 60% 40%;
    animation: rotation var(--time-animation) linear infinite;
  }

  .loader svg #clipping polygon:nth-child(7) {
    transform-origin: 60% 40%;
    animation: rotation var(--time-animation) linear infinite;
    animation-delay: calc(var(--time-animation) / -1.5);
  }

  @keyframes rotation {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  @keyframes roundness {
    0% {
      filter: contrast(15);
    }
    20% {
      filter: contrast(3);
    }
    40% {
      filter: contrast(3);
    }
    60% {
      filter: contrast(15);
    }
    100% {
      filter: contrast(15);
    }
  }

  @keyframes colorize {
    0% {
      filter: hue-rotate(0deg);
    }
    20% {
      filter: hue-rotate(-30deg);
    }
    40% {
      filter: hue-rotate(-60deg);
    }
    60% {
      filter: hue-rotate(-90deg);
    }
    80% {
      filter: hue-rotate(-45deg);
    }
    100% {
      filter: hue-rotate(0deg);
    }
  }

  .orb-container {
    position: relative;
    width: 100px;
    height: 100px;
    margin-bottom: 12px;
  }

  .emoji-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 2.2rem;
    z-index: 10;
    pointer-events: none;
    animation: float 2.5s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% { transform: translate(-50%, -50%) translateY(0px); }
    50% { transform: translate(-50%, -50%) translateY(-5px); }
  }

  .greeting-text {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
    text-align: center;
  }

  .subtitle-text {
    font-size: 0.85rem;
    color: #94a3b8;
    text-align: center;
  }
</style>
</head>
<body>
  <div class="orb-container">
    <div class="loader">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <defs>
          <mask id="clipping">
            <polygon points="0,0 100,0 100,100 0,100" fill="black"></polygon>
            <polygon points="25,25 75,25 50,75" fill="white"></polygon>
            <polygon points="50,25 75,75 25,75" fill="white"></polygon>
            <polygon points="35,35 65,35 50,65" fill="white"></polygon>
            <polygon points="35,35 65,35 50,65" fill="white"></polygon>
            <polygon points="35,35 65,35 50,65" fill="white"></polygon>
            <polygon points="35,35 65,35 50,65" fill="white"></polygon>
          </mask>
        </defs>
      </svg>
      <div class="box"></div>
    </div>
    <div class="emoji-overlay">≡ƒÄô</div>
  </div>
  <div class="greeting-text">GREETING_VAL, <span style="color:#60a5fa;">NAME_VAL</span>! ≡ƒæï</div>
  <div class="subtitle-text">How can EduSphere AI help you today?</div>
</body>
</html>"""

        _html = _html.replace("GREETING_VAL", _greeting_v).replace("NAME_VAL", _name_v)
        _stc_orb.html(_html, height=210, scrolling=False)
        _card_close()


        # Quick ask
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### ≡ƒÆ¼ Quick Ask EduSphere AI")
        quick_q = st.text_area(
            "Ask me anything...",
            placeholder="Ask me anything ΓÇö e.g. 'Explain Newton's 3rd law', 'Write a Python function...'",
            height=80,
            key="dashboard_quick_ask",
            label_visibility="collapsed"
        )
        qcol1, qcol2 = st.columns([1, 4])
        with qcol1:
            ask_btn = st.button("Γû╢ Ask", key="dashboard_ask_btn", use_container_width=True)
        with qcol2:
            st.caption("Powered by Groq LLaMA 3.1 ┬╖ RAG Studio")

        if ask_btn and quick_q.strip():
            with logo_spinner("Thinking..."):
                answer = groq_chat(
                    quick_q.strip(),
                    system="You are EduSphere AI, a helpful educational assistant. Give a concise but thorough answer."
                )
                st.session_state.total_queries = total_q + 1
            st.markdown(f"**≡ƒñû EduSphere AI:**\n\n{answer}")
        _card_close()

        # Quick-access module cards
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### ≡ƒÜÇ Quick Module Access")
        modules_grid = [
            ("≡ƒô¥", "Summarise PDF", "≡ƒô¥ Executive Summariser", "#00f0ff"),
            ("≡ƒÄ¿", "Generate Image", "≡ƒÄ¿ AI Image Generator", "#8b5cf6"),
            ("≡ƒÆ╗", "Write Code", "≡ƒÆ╗ Code Lab & Explainer", "#4ade80"),
            ("≡ƒîì", "Translate Text", "≡ƒîì Academic Translator", "#fb923c"),
            ("≡ƒôè", "Analyse Data", "≡ƒôè System Analytics", "#f472b6"),
            ("≡ƒº¬", "Take Quiz", "≡ƒº¬ Quiz & Assessment Generator", "#38bdf8"),
        ]
        mcols = st.columns(3)
        for idx, (icon, label, nav_key, color) in enumerate(modules_grid):
            with mcols[idx % 3]:
                # Render beautiful custom cards with a real Streamlit overlay button
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                    border-radius:12px; padding:14px 12px; text-align:center; position:relative;
                                    transition:all 0.2s; min-height: 125px;">
                            <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
                            <div style="font-size:0.78rem; font-weight:600; color:var(--text); margin-bottom:4px;">{label}</div>
                            <div style="font-size:0.65rem; color:{color}; font-weight:700;">{nav_key.split(' ', 1)[1]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Overlay button to make the card clickable and trigger a sidebar state change
                    if st.button(f"≡ƒÜÇ Open {label}", key=f"quick_btn_{idx}", use_container_width=True):
                        st.session_state.selected_menu = nav_key
                        st.rerun()
                    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    with right_col:
        # Memory / Vector Index Status
        _card_open()
        st.markdown("#### ≡ƒùâ∩╕Å Memory Status")
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
        st.markdown("#### ≡ƒÆ¼ Recent Chats")
        history = st.session_state.get("chat_history", [])
        if history:
            recents = [m for m in history if m["role"] == "user"][-4:][::-1]
            for msg in recents:
                preview = msg["msg"][:42] + "..." if len(msg["msg"]) > 42 else msg["msg"]
                st.markdown(
                    f"""
                    <div style="padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.05);
                                font-size:0.78rem; color:var(--sub); display:flex; justify-content:space-between;">
                        <span style="color:var(--text);">≡ƒÆ¼ {preview}</span>
                        <span style="font-size:0.68rem; flex-shrink:0; margin-left:8px;">now</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No chat history yet ΓÇö start a conversation in EduChat!")
        _card_close()

        # Model selector (display only)
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### ≡ƒñû Active Model")
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
            f'<div style="font-size:0.72rem; color:#4ade80; margin-top:4px;">ΓùÅ Online ┬╖ Groq Accelerated</div>',
            unsafe_allow_html=True
        )
        _card_close()

        # EduSphere info card (kept!)
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        _card_open()
        st.markdown("#### ≡ƒÄô EduSphere AI")
        st.markdown(
            """
            <div style="font-size:0.78rem; color:var(--sub); line-height:1.7;">
                <div style="color:var(--accent); font-weight:600; margin-bottom:6px;">
                    Enterprise Educational Ecosystem
                </div>
                <div>ΓùÅ 14 AI-powered modules</div>
                <div>ΓùÅ FAISS RAG document search</div>
                <div>ΓùÅ 7 premium UI themes</div>
                <div>ΓùÅ Groq LLaMA 3.1 inference</div>
                <div>ΓùÅ 3D CesiumJS globe</div>
                <div style="margin-top:8px; color:var(--accent2); font-size:0.7rem;">
                    Powered by Groq ┬╖ FAISS ┬╖ Streamlit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        _card_close()


