# 🚀 JSON-Only LLM Task Runner (Gemini Edition)

A FastAPI microservice that generates, uploads, and publishes lightweight front-end projects automatically using Google’s **Gemini API** and GitHub.

It replaces manual repo setup and file generation by orchestrating:
- Gemini-powered HTML/CSS/JS generation  
- Automatic GitHub repo creation and file uploads  
- Auto-deployment to GitHub Pages  
- Optional callback reporting for evaluations or grading systems

---

## 🧠 How It Works

1. **POST** a JSON task to `/upload-task` with:
   - `task`: a short task name  
   - `brief`: the user’s short project description  
   - `email`: optional for tracking  
   - `round`: optional round number (defaults to 1)  
   - `nonce`: unique ID or timestamp for repo name  
   - `attachments`: optional base64 files to upload  
   - `evaluation_url`: optional callback URL  
   - `secret`: required if `SUBMISSION_SECRET` is set in `.env`

2. The server will:
   - Create (or reuse) a GitHub repository named `{task}-{nonce}`
   - Use Gemini to generate:
     - `index.html`
     - `styles.css`
     - `script.js`
   - Upload them to `main` or `round-n` branch
   - Create/refresh a `gh-pages` branch for live preview
   - Generate a professional `README.md` using Gemini  
   - Post back status JSON to your evaluation callback (if provided)

---

## 🧩 Environment Variables (`.env`)

```bash
# --- Gemini API ---
GEMINI_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta

# --- GitHub ---
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_OWNER=your_github_username_or_org

# --- Optional secrets ---
SUBMISSION_SECRET=optional_secret_string
