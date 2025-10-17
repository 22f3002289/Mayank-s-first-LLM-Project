import os
import json
import time
import base64
import re
from typing import Optional, Dict, Any
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")  # optional: org or username

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. LLM calls will fail until configured.")

app = FastAPI(title="JSON-only LLM Task Runner (Gemini Edition)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Gemini client ----------
# ---------- Gemini client (robust) ----------
class GeminiClient:
    def __init__(self, key: Optional[str], model: str = GEMINI_MODEL, api_base: str = GEMINI_API_BASE):
        self.key = key
        self.model = model
        self.api_base = api_base.rstrip("/")

    def _extract_text_from_response(self, j: Dict[str, Any]) -> Optional[str]:
        # Common path used by many examples
        try:
            cand = j.get("candidates", [])
            if cand and isinstance(cand, list):
                c0 = cand[0]
                # new shape: content.parts is list of strings or objects
                cont = c0.get("content", {})
                if isinstance(cont, dict):
                    # case: content.parts -> list of strings or {"text": "..."}
                    parts = cont.get("parts") or cont.get("text") or cont.get("output")
                    if isinstance(parts, list) and parts:
                        # join string parts or extract 'text' fields
                        texts = []
                        for p in parts:
                            if isinstance(p, str):
                                texts.append(p)
                            elif isinstance(p, dict):
                                # some SDKs use { "text": "..."} or { "role": "...", "text": "..." }
                                if "text" in p and isinstance(p["text"], str):
                                    texts.append(p["text"])
                                else:
                                    # fallback to stringify dict
                                    texts.append(json.dumps(p))
                        if texts:
                            return "\n".join(texts).strip()
                    elif isinstance(parts, str):
                        return parts.strip()
                    # another possibility: content.parts is a single string inside nested structure
                    # or candidate content is directly a dict with string fields — try to find first string leaf
                # if cont is a string directly
                if isinstance(cont, str) and cont.strip():
                    return cont.strip()
        except Exception:
            pass

        # Generic DFS: find the first sufficiently long string leaf
        def find_string_leaf(x):
            if isinstance(x, str) and len(x.strip()) > 10:
                return x.strip()
            if isinstance(x, list):
                for el in x:
                    res = find_string_leaf(el)
                    if res:
                        return res
            if isinstance(x, dict):
                # prioritize common keys
                for k in ("text", "parts", "content", "output"):
                    if k in x:
                        res = find_string_leaf(x[k])
                        if res:
                            return res
                for v in x.values():
                    res = find_string_leaf(v)
                    if res:
                        return res
            return None

        maybe = find_string_leaf(j)
        if maybe:
            return maybe.strip()
        return None

    async def chat(self, system_prompt: str, user_prompt: str, timeout: float = 60.0, max_tokens: int = 1500) -> str:
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY not configured")

        url = f"{self.api_base}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.key}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
            if 200 <= r.status_code < 300:
                j = r.json()
                text = self._extract_text_from_response(j)
                if text:
                    return text
                # no text found — return pretty JSON so caller can log it but avoid storing raw JSON in content files
                return json.dumps(j, indent=2)
            else:
                raise RuntimeError(f"Gemini API error {r.status_code}: {r.text}")

gemini_client = GeminiClient(GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_BASE)

# ---------- GitHub helpers ----------
GITHUB_API = "https://api.github.com"

async def _gh_request(method: str, url: str, **kwargs) -> httpx.Response:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set.")
    headers = kwargs.pop("headers", {})
    headers.update({"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"})
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, headers=headers, **kwargs)
        return resp

async def github_create_repo(name: str, description: str = "", private: bool = False) -> Dict[str, Any]:
    payload = {"name": name, "description": description, "private": private, "auto_init": False}
    # prefer org creation if GITHUB_OWNER is set
    if GITHUB_OWNER:
        url = f"{GITHUB_API}/orgs/{GITHUB_OWNER}/repos"
        r = await _gh_request("POST", url, json=payload)
        if r.status_code == 201:
            return r.json()
        # fallthrough to user creation
    url_user = f"{GITHUB_API}/user/repos"
    r2 = await _gh_request("POST", url_user, json=payload)
    if r2.status_code == 201:
        return r2.json()
    # include token owner debug if possible
    me = await _gh_request("GET", f"{GITHUB_API}/user")
    me_login = me.json().get("login") if me.status_code == 200 else "unknown"
    raise RuntimeError(f"Failed to create repo as {me_login}: {r2.status_code} {r2.text}")

async def github_get_ref(owner: str, repo: str, ref: str):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/refs/heads/{ref}"
    r = await _gh_request("GET", url)
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(f"Failed to get ref {ref}: {r.status_code} {r.text}")

async def github_create_ref(owner: str, repo: str, ref: str, sha: str):
    payload = {"ref": ref, "sha": sha}
    r = await _gh_request("POST", f"{GITHUB_API}/repos/{owner}/{repo}/git/refs", json=payload)
    if r.status_code in (200, 201):
        return r.json()
    raise RuntimeError(f"Failed to create ref {ref}: {r.status_code} {r.text}")

async def github_create_file(owner: str, repo: str, path: str, content_bytes: bytes, message: str, branch: str = "main"):
    """
    Create or update file on branch using Contents API.
    If file exists, fetch its sha and include it to update.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {"message": message, "content": b64, "branch": branch}
    async with httpx.AsyncClient() as client:
        get_r = await client.get(f"{url}?ref={branch}", headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"})
        if get_r.status_code == 200:
            try:
                sha = get_r.json().get("sha")
                if sha:
                    payload["sha"] = sha
            except Exception:
                pass
        put_r = await client.put(url, json=payload, headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"})
        if put_r.status_code in (200, 201):
            return put_r.json()
        else:
            raise RuntimeError(f"Failed to create/update file {path} on branch {branch}: {put_r.status_code} {put_r.text}")

# ---------- Fallback templates ----------
FALLBACK_INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Auto Generated Page</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main class="centered">
    <div class="card">
      <h1>Auto Generated</h1>
      <p>This page was generated based on the brief provided.</p>
      <form id="demoForm">
        <input name="email" placeholder="Email" type="email" required/>
        <input name="password" placeholder="Password" type="password" required/>
        <button type="submit">Submit</button>
      </form>
      <button id="toggleTheme">Toggle Dark</button>
      <div id="out"></div>
    </div>
  </main>
  <script src="script.js"></script>
</body>
</html>"""

FALLBACK_CSS = """:root{
  --bg:#ffffff; --card:#ffffff; --text:#111; --muted:#666; --accent:#0b84ff;
}
body{background:var(--bg); color:var(--text); font-family:Arial, Helvetica, sans-serif; margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;}
.centered{width:100%; max-width:480px; padding:20px;}
.card{background:var(--card); padding:24px; border-radius:12px; box-shadow:0 8px 30px rgba(16,24,40,0.08);}
input{display:block; width:100%; padding:10px; margin-bottom:10px; border-radius:8px; border:1px solid #ddd;}
button{background:var(--accent); color:white; border:0; padding:10px 14px; border-radius:8px;}
[data-theme='dark']{ --bg:#0b0d11; --card:#0f1114; --text:#e6eef8; --muted:#9aa8bb; --accent:#4ea3ff;}
"""

FALLBACK_JS = """document.getElementById('demoForm').addEventListener('submit', function(e){
  e.preventDefault();
  const fd = new FormData(e.target);
  const out = {email: fd.get('email'), password: fd.get('password')};
  document.getElementById('out').innerText = 'Demo submit: ' + JSON.stringify(out);
});
document.getElementById('toggleTheme').addEventListener('click', function(){
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
});
"""

# ---------- LLM generation (marker-based) ----------
async def generate_project_from_brief(brief: str, task_name: str) -> Dict[str, bytes]:
    """
    Ask the LLM to produce three files using clear markers:
      ---index.html---
      ---styles.css---
      ---script.js---
    Returns mapping filename->bytes. Falls back to deterministic templates on parse failure.
    """
    system_prompt = (
        "You are a senior front-end engineer. Given a short brief, produce a minimal but complete "
        "front-end project. Output exactly three sections in plain text using these delimiters:\n"
        "---index.html---\n(HTML content)\n---styles.css---\n(CSS content)\n---script.js---\n(JS content)\n\n"
        "Do NOT output JSON, do NOT wrap in markdown fences. Keep files small and self-contained."
    )
    user_prompt = f"Task: {task_name}\nBrief: {brief}\nProduce the files as described."
    raw = await gemini_client.chat(system_prompt, user_prompt, timeout=60.0, max_tokens=2000)

    # Parse marker-separated output
    try:
        parts = re.split(r"---(index\.html|styles\.css|script\.js)---", raw)
        files: Dict[str, bytes] = {}
        current = None
        for p in parts:
            if p is None:
                continue
            p = p.strip()
            if p in ("index.html", "styles.css", "script.js"):
                current = p
                files[current] = b""
            elif current:
                files[current] = (files[current] + ("\n" + p).encode("utf-8")).lstrip(b"\n")
        if "index.html" in files and files["index.html"].strip():
            files.setdefault("styles.css", FALLBACK_CSS.encode("utf-8"))
            files.setdefault("script.js", FALLBACK_JS.encode("utf-8"))
            return files
    except Exception:
        pass

    # If parsing failed, try to extract JSON mapping (some models produce JSON)
    try:
        txt = raw.strip()
        txt = re.sub(r"^```(?:\w+)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            candidate = m.group(0)
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                out: Dict[str, bytes] = {}
                for k, v in parsed.items():
                    if k in ("index.html", "styles.css", "script.js") and isinstance(v, str):
                        out[k] = v.encode("utf-8")
                if out:
                    out.setdefault("index.html", FALLBACK_INDEX.encode("utf-8"))
                    out.setdefault("styles.css", FALLBACK_CSS.encode("utf-8"))
                    out.setdefault("script.js", FALLBACK_JS.encode("utf-8"))
                    return out
    except Exception:
        pass

    # Deterministic fallback
    return {
        "index.html": FALLBACK_INDEX.encode("utf-8"),
        "styles.css": FALLBACK_CSS.encode("utf-8"),
        "script.js": FALLBACK_JS.encode("utf-8"),
    }

# ---------- Core handler ----------
async def handle_task_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Reuse repo per task: repo_name = {task}-{nonce}
    - Use branches per round: main or round-{n}
    - Upload generated or fallback files, create gh-pages, update README
    """
    email = data.get("email")
    task_name = data.get("task", "task")
    round_no = int(data.get("round", 1) or 1)
    nonce = data.get("nonce", str(int(time.time())))
    brief = data.get("brief", "")
    evaluation_url = data.get("evaluation_url")

    base_repo_name = f"{task_name}-{nonce}".lower().replace(" ", "-")
    repo_name = base_repo_name
    report: Dict[str, Any] = {"status": "pending", "repo": None, "pages_url": None, "errors": [], "llm_files": [], "attachments_uploaded": []}

    # --- 1) Check if repo exists; if not, create it (reuse across rounds) ---
    owner = None
    existing_repo = None
    try:
        # determine owner for repo check: use configured GITHUB_OWNER or authenticated user
        if GITHUB_OWNER:
            owner_to_check = GITHUB_OWNER
        else:
            me = await _gh_request("GET", f"{GITHUB_API}/user")
            if me.status_code == 200:
                owner_to_check = me.json().get("login")
            else:
                owner_to_check = None

        if owner_to_check:
            r_check = await _gh_request("GET", f"{GITHUB_API}/repos/{owner_to_check}/{repo_name}")
            if r_check.status_code == 200:
                existing_repo = r_check.json()
                owner = existing_repo["owner"]["login"]
                report["repo"] = existing_repo.get("html_url")
    except Exception:
        existing_repo = None

    if not existing_repo:
        try:
            repo_json = await github_create_repo(repo_name, description=brief, private=False)
            report["repo"] = repo_json.get("html_url")
            owner = repo_json["owner"]["login"]
        except Exception as e:
            report["errors"].append(f"repo_create_failed:{e}")
            # best-effort notify evaluation_url
            if evaluation_url:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(evaluation_url, json={"status": "repo_create_failed", "details": report})
                except Exception:
                    pass
            return report

    # --- 2) Add LICENSE and handle attachments ---
    try:
        year = time.gmtime().tm_year
        license_text = f"MIT License\n\nCopyright (c) {year} {owner}\n\nPermission is hereby granted..."
        await github_create_file(owner, repo_name, "LICENSE", license_text.encode("utf-8"), "Add LICENSE", branch="main")
    except Exception as e:
        report["errors"].append(f"license_failed:{e}")

    attachments = data.get("attachments", [])
    if isinstance(attachments, list):
        for att in attachments:
            if isinstance(att, dict):
                att_name = att.get("name") or "sample.png"
                att_url = att.get("url") or att.get("data") or ""
                if isinstance(att_url, str) and att_url.startswith("data:"):
                    m = re.match(r"data:(?P<mime>[^;]+);base64,(?P<b64>.+)", att_url)
                    if m:
                        b64 = m.group("b64")
                        try:
                            content_bytes = base64.b64decode(b64)
                        except Exception as e:
                            report["errors"].append(f"attachment_base64_decode_failed:{att_name}:{e}")
                            continue
                        try:
                            await github_create_file(owner, repo_name, att_name, content_bytes, f"Add {att_name}", branch="main")
                            report["attachments_uploaded"].append({"name": att_name, "branch": "main"})
                        except Exception as e:
                            report["errors"].append(f"attachment_main_failed:{att_name}:{e}")
                        try:
                            await github_create_file(owner, repo_name, att_name, content_bytes, f"Add {att_name} for pages", branch="gh-pages")
                            report["attachments_uploaded"].append({"name": att_name, "branch": "gh-pages"})
                        except Exception as e:
                            report["errors"].append(f"attachment_pages_failed:{att_name}:{e}")
                    else:
                        report["errors"].append(f"attachment_malformed:{att_name}")

    # --- 3) LLM generate files and upload to branch for this round ---
    try:
        generated = await generate_project_from_brief(brief or f"Task: {task_name}", task_name)
        target_branch = "main" if round_no == 1 else f"round-{round_no}"
        # ensure branch exists if not main: create from main if needed
        if target_branch != "main":
            try:
                ref_main = await github_get_ref(owner, repo_name, "main")
                sha_main = ref_main["object"]["sha"]
                try:
                    await github_get_ref(owner, repo_name, target_branch)
                except Exception:
                    await github_create_ref(owner, repo_name, f"refs/heads/{target_branch}", sha_main)
            except Exception:
                pass

        for fname, content in generated.items():
            try:
                await github_create_file(owner, repo_name, fname, content, f"Add {fname} from LLM", branch=target_branch)
                report["llm_files"].append({"name": fname, "branch": target_branch})
            except Exception as e:
                report["errors"].append(f"llm_file_upload_failed:{fname}:{e}")
    except Exception as e:
        report["errors"].append(f"llm_generation_failed:{e}")

    # --- 4) Create gh-pages branch from main and put index.html there ---
    try:
        try:
            ref_main = await github_get_ref(owner, repo_name, "main")
            sha = ref_main["object"]["sha"]
        except Exception:
            sha = None

        if sha:
            try:
                await github_get_ref(owner, repo_name, "gh-pages")
            except Exception:
                try:
                    await github_create_ref(owner, repo_name, "refs/heads/gh-pages", sha)
                except Exception as ex:
                    report["errors"].append(f"gh_pages_ref_create_failed:{ex}")
            try:
                # try to fetch index.html from the branch we uploaded to (prefer the target branch)
                content = None
                try:
                    r = await _gh_request("GET", f"{GITHUB_API}/repos/{owner}/{repo_name}/contents/index.html?ref={target_branch}")
                    if r.status_code == 200:
                        content_b64 = r.json().get("content", "")
                        content = base64.b64decode(content_b64.encode("utf-8"))
                except Exception:
                    content = None
                if not content:
                    # fallback to generated mapping
                    if "index.html" in generated:
                        content = generated["index.html"]
                    else:
                        content = FALLBACK_INDEX.encode("utf-8")
                await github_create_file(owner, repo_name, "index.html", content, "Add index.html for gh-pages", branch="gh-pages")
                report["pages_url"] = f"https://{owner}.github.io/{repo_name}/"
                report.setdefault("checks", {})["pages_created"] = True
            except Exception as e:
                report["errors"].append(f"gh_pages_failed:{e}")
                report.setdefault("checks", {})["pages_created"] = False
        else:
            report["errors"].append("gh_pages_failed:main_missing")
            report.setdefault("checks", {})["pages_created"] = False
    except Exception as e:
        report["errors"].append(f"gh_pages_flow_failed:{e}")

    # --- 5) Polished README via LLM (safe update) ---
    try:
        system = "You are an assistant that writes concise README files for small demo repos."
        user = f"Write a short professional README describing: {brief}\nInclude usage instructions and files created."
        readme_text = await gemini_client.chat(system, user, timeout=60.0, max_tokens=800)
        if readme_text:
            try:
                await github_create_file(owner, repo_name, "README.md", readme_text.encode("utf-8"), "Update README via LLM", branch="main")
                report.setdefault("checks", {})["readme_generated"] = True
            except Exception as e:
                report["errors"].append(f"readme_upload_failed:{e}")
                report.setdefault("checks", {})["readme_generated"] = False
    except Exception as e:
        report["errors"].append(f"openai_readme_failed:{e}")
        report.setdefault("checks", {})["readme_generated"] = False

    # --- 6) Evaluation callback (best-effort) ---
    try:
        final_report = {
            "email": email,
            "task": task_name,
            "round": round_no,
            "repo": report.get("repo"),
            "pages_url": report.get("pages_url"),
            "checks": report.get("checks", {}),
            "errors": report.get("errors", []),
            "llm_files": report.get("llm_files", []),
            "attachments_uploaded": report.get("attachments_uploaded", []),
            "timestamp": int(time.time())
        }
        if evaluation_url:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(evaluation_url, json=final_report)
                report["evaluation_posted"] = (200 <= r.status_code < 300)
                report["evaluation_status_code"] = r.status_code
    except Exception as e:
        report["errors"].append(f"evaluation_post_failed:{e}")
        report["evaluation_posted"] = False

    report["status"] = "done" if not report["errors"] else "done_with_errors"
    return report

# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {"status": "ready", "note": "POST application/json to /upload-task with the task JSON body."}

@app.post("/upload-task")
async def upload_task(body: Dict[str, Any] = Body(...)):
    if not isinstance(body, dict):
        return JSONResponse(status_code=200, content={"status": "failed", "errors": ["body must be a JSON object"]})
    # --- secret checkpoint ---
    # Support either SUBMISSION_SECRET or STUDENT_SECRET in the environment.
    # If either is set, require the incoming JSON to include a matching `secret` field.
    submission_secret = os.getenv("SUBMISSION_SECRET") or os.getenv("STUDENT_SECRET")
    if submission_secret:
        supplied = body.get("secret")
        # normalize and compare trimmed string values to avoid whitespace mismatches
        if supplied is None or str(supplied).strip() != str(submission_secret).strip():
            # Return explicit secret mismatch error
            return JSONResponse(status_code=401, content={"status": "failed", "errors": ["secret mismatch"]})
    try:
        report = await handle_task_from_dict(body)
        return JSONResponse(status_code=200, content=report)
    except Exception as e:
        return JSONResponse(status_code=200, content={"status": "failed", "errors": [str(e)]})

@app.get("/solve")
async def solve(url: str):
    """
    Demo solver: fetch image at `url` and ask LLM to extract text (text-only LLM OCR).
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch url: {r.status_code}")
            img_bytes = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"fetch error: {e}")

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    system_prompt = "You are an assistant that extracts short textual captchas from base64 images. Reply only with the text or ERROR:UNREADABLE."
    user_prompt = f"Below is a base64-encoded image. Try to read any textual characters. If unreadable, reply EXACTLY: ERROR:UNREADABLE.\n\nIMAGE_BASE64_START\n{b64}\nIMAGE_BASE64_END\n\nReply ONLY with the extracted text."
    try:
        solved = await gemini_client.chat(system_prompt, user_prompt, timeout=40.0, max_tokens=128)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return {"solved_text": solved.strip()}
