# EduSphere AI

> **Enterprise-grade, privacy-first educational AI platform** powered by GROQ LLaMA 3.1, FAISS RAG, and Streamlit.

---

## ✨ Features

| Module | Description |
|---|---|
| 🧠 EduChat & RAG Studio | Document-aware chatbot with FAISS vector retrieval |
| 📚 Study Planner | AI-generated personalised study schedules |
| 🔬 Socratic Clarifier | Concept deconstruction through Socratic dialogue |
| 🧪 Quiz Generator | Auto-generated MCQ/True-False assessments |
| 💻 Code Lab | Bug detection, explanation, and complexity optimisation |
| 🌍 Academic Translator | Multi-lingual translation with formality control |
| 📝 Executive Summariser | Bullet/abstract/ELI5 text summarisation |
| 🖼️ URL Intelligence | Web page scraping and AI summarisation |
| 🧹 Background Remover | AI-powered image background removal |
| 📊 Analytics | Session telemetry and usage dashboard |
| 🏛️ Architecture | System architecture and DB schema documentation |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- A [GROQ API key](https://console.groq.com/) (free tier available)

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd edusphere-ai

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 4. Run the App

```bash
streamlit run src/main.py
```

Open your browser at `http://localhost:8501`.

**Demo credentials:**

| Role | Email | Password |
|---|---|---|
| Student | `student@edusphere.ai` | `student123` |
| Admin | `admin@edusphere.ai` | `admin123` |

---

## 🗂️ Project Structure

```text
edusphere-ai/
├── .env                    # API keys (git-ignored)
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py             # Streamlit entrypoint
│   ├── auth.py             # bcrypt authentication
│   ├── config.py           # constants & logging
│   ├── exceptions.py       # custom exception hierarchy
│   ├── modules.py          # all 11 feature modules
│   ├── rag.py              # FAISS RAG pipeline
│   └── utils.py            # GROQ client & helpers
├── assets/
│   └── styles.css          # global stylesheet
├── tests/
│   ├── test_auth.py
│   ├── test_rag.py
│   └── test_utils.py
└── logs/
    └── app.log             # structured application logs
```

---

## 🧪 Running Tests

```bash
pytest -v --cov=src tests/
```

---

## 🔒 Security

- Passwords are hashed with **bcrypt** (cost factor 12) — never stored as plain text.
- All API keys are loaded from environment variables, never hard-coded.
- Multi-layer guardrails detect and block harmful, private, and crisis inputs **before** reaching the LLM.
- Structured logs never contain passwords, API keys, or PII.

---

## 📄 License

MIT © EduSphere AI Team
