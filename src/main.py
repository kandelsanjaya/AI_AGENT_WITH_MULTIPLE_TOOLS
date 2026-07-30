"""
main.py
=======
EduSphere AI — Main Streamlit Application Entry Point.
Enhanced with LICT Campus Assistant integration, SQLite persistence,
12 feature modules, and multi-format export capabilities.

Run with:
    streamlit run DasaAI.py

Environment:
    GROQ_API_KEY   — required, set in .env
    TAVILY_API_KEY — optional, for live web search
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (MUST be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduSphere AI — Enterprise Learning Ecosystem",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Internal imports (after set_page_config)
# ---------------------------------------------------------------------------
from src.auth import get_user_info, verify_credentials, update_user_credentials  # noqa: E402
from src.config import DEFAULT_THEME, GROQ_API_KEY, THEMES, AVAILABLE_MODELS  # noqa: E402
from src.database import init_db  # noqa: E402
from src.modules import (  # noqa: E402
    render_analytics,
    render_architecture,
    render_bg_remover,
    render_code_lab,
    render_educhat,
    render_quiz_generator,
    render_resume_builder,
    render_socratic_clarifier,
    render_study_planner,
    render_summariser,
    render_translator,
    render_url_intelligence,
)

# ---------------------------------------------------------------------------
# Initialise SQLite database
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------

def inject_css(theme: dict) -> None:
    """Load the external CSS file and inject CSS custom properties for theming."""
    css_path = Path(__file__).resolve().parent.parent / "assets" / "styles.css"
    if css_path.exists():
        raw_css = css_path.read_text(encoding="utf-8")
    else:
        raw_css = ""

    theme_vars = f"""
    :root {{
        --bg:          {theme['bg']};
        --card:        {theme['card']};
        --accent:      {theme['accent']};
        --accent2:     {theme['accent2']};
        --text:        {theme['text']};
        --sub:         {theme['sub']};
        --glow:        {theme['glow']};
        --border-neon: {theme['border_neon']};
    }}
    """

    # Background styling for login screen
    bg_style = ""
    if not st.session_state.get("authenticated"):
        map_path = Path(__file__).resolve().parent.parent / "assets" / "nepal_map.png"
        if map_path.exists():
            import base64
            b64_str = base64.b64encode(map_path.read_bytes()).decode("utf-8")
            bg_style = f"""
            .stApp {{
                background-image: linear-gradient(rgba(11, 17, 32, 0.4), rgba(11, 17, 32, 0.45)), url("data:image/png;base64,{b64_str}") !important;
                background-size: cover, contain !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
            }}
            .login-box {{
                background: rgba(255, 255, 255, 0.04) !important;
                backdrop-filter: blur(30px) saturate(190%) !important;
                -webkit-backdrop-filter: blur(30px) saturate(190%) !important;
                border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5) !important;
            }}
            """

    st.markdown(f"<style>{theme_vars}\n{bg_style}\n{raw_css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_info": None,
        "chat_history": [],
        "session_start": datetime.datetime.now(),
        "total_queries": 0,
        "blocked_count": 0,
        "theme": DEFAULT_THEME,
        "vector_index": None,  # VectorIndex | None
        "generated_resume": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ---------------------------------------------------------------------------
# Authentication Screen
# ---------------------------------------------------------------------------

def render_login_screen(theme: dict) -> None:
    """Render the login UI with centered branding on top, and form/credentials side-by-side below."""
    # Row 1: Centered Branding Card
    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        st.markdown(
            f"""
            <div class="login-box" style="margin-top: 10px; text-align: center; max-width: 100%; margin-bottom: 20px;">
                <div class="scanner-container" style="margin-top: 15px;">
                    <div class="scanner-eye left-eye"><div class="scanner-pupil"></div></div>
                    <div class="scanner-eye right-eye"><div class="scanner-pupil"></div></div>
                </div>
                <div class="navbar-title" style="font-size: 2.1rem; margin-top: 15px;">EDUSPHERE AI</div>
                <div style="color:var(--sub); font-size: 0.9rem; margin-top: 8px; margin-bottom: 5px;">
                    Enterprise Educational Platform & College AI Assistant
                </div>
                <div style="color:var(--sub); font-size: 0.78rem; margin-bottom: 20px; opacity:0.7;">
                    Powered by Groq LLaMA 3.1 · FAISS RAG · 12 AI Modules
                </div>
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 15px; font-size: 0.8rem; color: var(--sub);">
                    ● 12 AI Agents Active &nbsp;&nbsp;&nbsp;&nbsp; ● 98.7% Accuracy &nbsp;&nbsp;&nbsp;&nbsp; ● College Knowledge Base
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2: Form & Credentials Side-by-Side
    col1, col2 = st.columns([1.1, 1], gap="medium")

    with col1:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### 🔒 Authenticate")
            email_input = st.text_input(
                "📧 Email / User ID",
                value="student@edusphere.ai",
                autocomplete="email",
            )
            password_input = st.text_input(
                "🔒 Password",
                type="password",
                value="",
                help="Enter your account password.",
            )
            submit = st.form_submit_button("🔑 Authenticate", use_container_width=True)

            if submit:
                if not email_input.strip() or not password_input:
                    st.warning("Please fill in both fields.")
                elif verify_credentials(email_input.strip(), password_input):
                    user = get_user_info(email_input.strip())
                    st.session_state.authenticated = True
                    st.session_state.user_info = user
                    st.session_state.session_start = datetime.datetime.now()
                    st.success("✅ Success! Loading...")
                    time.sleep(0.6)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

    with col2:
        st.info(
            "💡 **Demo Credentials**\n\n"
            "Use these details to log in:\n\n"
            "- **Student Role**\n"
            "  - Email: `student@edusphere.ai`\n"
            "  - Pass: `student123`\n\n"
            "- **Admin Role**\n"
            "  - Email: `admin@edusphere.ai`\n"
            "  - Pass: `admin123`"
        )


# ---------------------------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------------------------

NAV_OPTIONS = [
    "🧠 EduChat & RAG Studio",
    "📚 Study Planner & Syllabus",
    "🔬 Socratic Concept Clarifier",
    "🧪 Quiz & Assessment Generator",
    "💻 Code Lab & Explainer",
    "🌍 Academic Translator",
    "📝 Executive Summariser",
    "🖼️ Visual & URL Intelligence",
    "🧹 Background Remover",
    "📋 Resume Builder",
    "📊 System Analytics",
    "🏛️ Architecture Blueprint",
    "⚙️ Settings & Profile",
]

MODULE_MAP = {
    "🧠 EduChat & RAG Studio": render_educhat,
    "📚 Study Planner & Syllabus": render_study_planner,
    "🔬 Socratic Concept Clarifier": render_socratic_clarifier,
    "🧪 Quiz & Assessment Generator": render_quiz_generator,
    "💻 Code Lab & Explainer": render_code_lab,
    "🌍 Academic Translator": render_translator,
    "📝 Executive Summariser": render_summariser,
    "🖼️ Visual & URL Intelligence": render_url_intelligence,
    "🧹 Background Remover": render_bg_remover,
    "📋 Resume Builder": render_resume_builder,
    "📊 System Analytics": render_analytics,
    "🏛️ Architecture Blueprint": render_architecture,
    "⚙️ Settings & Profile": lambda: render_settings(),
}


def render_settings() -> None:
    """⚙️ Settings & Profile Manager — allow updating profile information, profile picture, and switching accounts."""
    st.markdown("### ⚙️ Settings & Profile Manager")

    user = st.session_state.user_info or {}

    col1, col2 = st.columns([1, 1])

    with col1:
        _card_open()
        st.markdown("#### 👤 Update Profile Information")
        with st.form("update_profile_form"):
            new_name = st.text_input("Full Name", value=user.get("name", ""))
            new_email = st.text_input("Email Address", value=user.get("email", ""))
            new_pwd = st.text_input("New Password", type="password", placeholder="Leave blank to keep current")
            submit_profile = st.form_submit_button("💾 Save Profile Updates")

            if submit_profile:
                if not new_name.strip() or not new_email.strip():
                    st.warning("Name and Email cannot be empty.")
                else:
                    try:
                        success = update_user_credentials(
                            current_email=user.get("email", ""),
                            new_email=new_email,
                            new_password=new_pwd if new_pwd.strip() else None,
                            new_name=new_name
                        )
                        if success:
                            st.session_state.user_info = {
                                "name": new_name,
                                "email": new_email,
                                "role": user.get("role", "Student")
                            }
                            st.success("✅ Profile updated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to update profile.")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")
        _card_close()

    with col2:
        _card_open()
        st.markdown("#### 🖼️ Upload Profile Picture")
        uploaded_pic = st.file_uploader("Choose an Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_pic:
            import base64
            bytes_data = uploaded_pic.read()
            b64_str = base64.b64encode(bytes_data).decode("utf-8")
            st.session_state.profile_pic = f"data:image/png;base64,{b64_str}"
            st.success("✅ Profile picture uploaded!")
            st.rerun()

        if st.session_state.get("profile_pic"):
            st.markdown("##### Current Profile Picture Preview")
            st.markdown(
                f'<div style="text-align:center; margin-top:10px;">'
                f'<img src="{st.session_state.profile_pic}" style="border-radius:50%; width:120px; height:120px; border:2px solid var(--border-neon); object-fit:cover; margin-bottom:10px; box-shadow:0 0 15px var(--glow);"/>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button("🗑️ Remove Picture"):
                st.session_state.profile_pic = None
                st.success("Profile picture removed.")
                st.rerun()
        _card_close()


def _card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="g-card" style="{extra_style}">', unsafe_allow_html=True)


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render the sidebar and return the selected menu item."""
    with st.sidebar:
        user = st.session_state.user_info or {}

        # Profile Picture & Name Display
        if st.session_state.get("profile_pic"):
            st.markdown(
                f'<div style="text-align:center; margin-bottom:15px;">'
                f'<img src="{st.session_state.profile_pic}" style="border-radius:50%; width:80px; height:80px; border:2px solid var(--border-neon); object-fit:cover; box-shadow:0 0 10px var(--glow);"/>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:15px; font-size:3.5rem;">'
                f'👤'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(f"<h3 style='text-align:center; margin-bottom:5px; margin-top:0;'>{user.get('name', 'User')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:var(--sub); margin-top:0;'>`{user.get('role', 'Student')}`</p>", unsafe_allow_html=True)
        st.markdown("---")

        # Theme selector
        theme_keys = list(THEMES.keys())
        current_idx = theme_keys.index(st.session_state.theme) if st.session_state.theme in theme_keys else 0
        selected_theme = st.selectbox("🎨 Theme", theme_keys, index=current_idx)
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

        st.markdown("---")

        # Navigation
        selected = st.radio("📌 Navigation", NAV_OPTIONS, label_visibility="collapsed")

        st.markdown("---")

        # API key status
        if GROQ_API_KEY:
            st.success("🟢 GROQ API Connected")
        else:
            st.error("🔴 GROQ API Key Missing — add to .env")

        # Chat export section
        chat_history = st.session_state.get("chat_history", [])
        if chat_history:
            st.markdown("---")
            st.markdown("**📤 Export Chat**")
            from src.utils import export_chat_as_markdown, export_chat_as_json
            md_export = export_chat_as_markdown(chat_history, title="EduSphere AI Chat Export")
            json_export = export_chat_as_json(chat_history)
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.download_button("📄 .md", md_export, file_name="chat_export.md", use_container_width=True)
            with col_e2:
                st.download_button("🧾 .json", json_export, file_name="chat_export.json", use_container_width=True)

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authenticated", "user_info", "chat_history", "vector_index",
                        "total_queries", "blocked_count", "profile_pic", "generated_resume"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    return selected


# ---------------------------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------------------------

def render_navbar() -> None:
    now_str = datetime.datetime.now().strftime("%H:%M | %d %b %Y")
    st.markdown(
        f"""
        <div class="navbar">
            <div>
                <div class="navbar-title">🎓 EDUSPHERE AI PLATFORM</div>
                <div style="color:var(--sub); font-size:0.78rem; margin-top:4px;">
                    Adaptive Educational System · College AI Assistant · Powered by LLaMA 3.1
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <span class="status-online">ONLINE</span>
                <span style="color:var(--sub); font-family:'JetBrains Mono'; font-size:0.78rem;">
                    {now_str}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    # Reset theme to default if the cached session theme was removed
    if st.session_state.theme not in THEMES:
        st.session_state.theme = DEFAULT_THEME

    # Always resolve the active theme first (needed for CSS)
    theme = THEMES[st.session_state.theme]
    inject_css(theme)

    if not st.session_state.authenticated:
        render_login_screen(theme)
        st.stop()

    selected_menu = render_sidebar()
    render_navbar()

    # Dispatch to the correct module
    render_fn = MODULE_MAP.get(selected_menu)
    if render_fn:
        render_fn()
    else:
        st.error(f"Unknown module: {selected_menu}")


if __name__ == "__main__":
    main()
