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
        "last_selected_menu": None,
        "active_tab": "login"
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ---------------------------------------------------------------------------
# Hamster Wheel Loader Helper
# ---------------------------------------------------------------------------

def render_hamster_loader(label: str = "EduSphere AI is running...", sub_label: str = "Spinning up resources"):
    """Injected Hamster loader animation using UIverse design."""
    st.markdown(
        f"""
        <div class="hamster-loader-wrapper">
            <div aria-label="Orange and tan hamster running in a metal wheel" role="img" class="wheel-and-hamster">
                <div class="wheel"></div>
                <div class="hamster">
                    <div class="hamster__body">
                        <div class="hamster__head">
                            <div class="hamster__ear"></div>
                            <div class="hamster__eye"></div>
                            <div class="hamster__nose"></div>
                        </div>
                        <div class="hamster__limb hamster__limb--fr"></div>
                        <div class="hamster__limb hamster__limb--fl"></div>
                        <div class="hamster__limb hamster__limb--br"></div>
                        <div class="hamster__limb hamster__limb--bl"></div>
                        <div class="hamster__tail"></div>
                    </div>
                </div>
                <div class="spoke"></div>
            </div>
            <div class="hamster-loader-label">{label}</div>
            <div class="hamster-loader-sub">{sub_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Authentication Screen
# ---------------------------------------------------------------------------

def render_login_screen(theme: dict) -> None:
    """Render the login & registration UI with tabs, social login support and custom CSS integrations."""
    # Row 1: Centered Single-Line Branding Header (Enhanced and more premium)
    st.markdown(
        f"""
        <div class="login-box" style="margin-top: 10px; max-width: 100%; margin-bottom: 25px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div class="scanner-container" style="margin-top: 5px; margin-bottom: 12px;">
                <div class="scanner-eye left-eye"><div class="scanner-pupil"></div></div>
                <div class="scanner-eye right-eye"><div class="scanner-pupil"></div></div>
            </div>
            <div class="navbar-title" style="font-size: 2.3rem; letter-spacing: 3px; font-weight: 900; margin-bottom: 8px;">EDUSPHERE AI</div>
            <div style="color:var(--text); font-size: 0.95rem; font-family: 'Space Grotesk', sans-serif; font-weight: 500; opacity: 0.95;">
                Enterprise Educational Ecosystem & Autonomous College AI Portal
            </div>
            <div style="color:var(--sub); font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; margin-top: 6px; opacity:0.8;">
                Groq LLaMA 3.1 &bull; Active FAISS RAG Storage &bull; 12 Integrated Agents
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 2: Tabs and Form/Credentials
    col1, col2 = st.columns([1.2, 1], gap="medium")

    with col1:
        # Custom HTML Tabs for clean looking layout
        t1, t2 = st.tabs(["🔑 Access Portal", "📝 Create Account"])

        with t1:
            # Login Form
            with st.form("login_form", clear_on_submit=False):
                st.markdown("#### 🔒 Login credentials")
                email_input = st.text_input(
                    "📧 Email or Username",
                    value="student@edusphere.ai",
                    autocomplete="email",
                    key="login_email"
                )
                password_input = st.text_input(
                    "🔒 Password",
                    type="password",
                    value="",
                    help="Enter your account password.",
                    key="login_password"
                )
                submit = st.form_submit_button("🔑 Authenticate Access", use_container_width=True)

                if submit:
                    if not email_input.strip() or not password_input:
                        st.warning("Please fill in both fields.")
                    elif verify_credentials(email_input.strip(), password_input):
                        user = get_user_info(email_input.strip())
                        st.session_state.authenticated = True
                        st.session_state.user_info = user
                        st.session_state.session_start = datetime.datetime.now()
                        st.success("✅ Success! Loading environment...")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")

            # Social logins section with Glassmorphic Round Buttons & SVG logos (Text-free)
            st.markdown('<div class="social-divider">OR AUTHENTICATE WITH</div>', unsafe_allow_html=True)
            
            # Use raw HTML with custom styling for social login row containing round buttons
            st.markdown(
                """
                <div class="social-login-container">
                    <button class="glass-social-btn google" onclick="alert('Google Auth integration coming soon!')">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" fill="#FBBC05"/>
                            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" fill="#EA4335"/>
                        </svg>
                    </button>
                    <button class="glass-social-btn apple" onclick="alert('Apple Auth integration coming soon!')">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M15.97 4.17c.66-.81 1.11-1.93.99-3.06-.96.04-2.13.64-2.82 1.45-.6.7-1.13 1.84-1.01 2.95.89.04 2.18-.53 2.84-1.34" fill="currentColor"/>
                        </svg>
                    </button>
                    <button class="glass-social-btn github" onclick="alert('GitHub Auth integration coming soon!')">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                        </svg>
                    </button>
                </div>
                """,
                unsafe_allow_html=True
            )

        with t2:
            # Registration Form
            with st.form("register_form", clear_on_submit=False):
                st.markdown("#### 📝 Register new account")
                reg_name = st.text_input("👤 Full Name", placeholder="e.g. John Doe")
                reg_email = st.text_input("📧 Email Address", placeholder="e.g. user@edusphere.ai")
                reg_pass = st.text_input("🔒 Password", type="password")
                reg_pass_conf = st.text_input("🔄 Confirm Password", type="password")
                reg_role = st.selectbox("🎓 Choose your Role", ["Student", "Professor", "Administrator"])
                
                reg_submit = st.form_submit_button("🚀 Register Account", use_container_width=True)

                if reg_submit:
                    if not reg_name.strip() or not reg_email.strip() or not reg_pass:
                        st.warning("Please fill all the fields.")
                    elif reg_pass != reg_pass_conf:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            register_user(
                                email=reg_email.strip(),
                                password=reg_pass,
                                name=reg_name.strip(),
                                role=reg_role
                            )
                            st.success("🎉 Account created successfully! Please log in above.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

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
