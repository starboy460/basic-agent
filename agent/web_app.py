from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import re
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zipfile import ZipFile

from agent.config import get_settings
from agent.config import PROJECT_ROOT
from agent.runner import AgentRunner
from agent.speech.adapters import WhisperSTTAdapter
from agent.speech.normalizer import normalize_for_tts


GENERATED_DIR = PROJECT_ROOT / "generated"
UPLOAD_DIR = PROJECT_ROOT / "uploads"


class WebAgent:
    def __init__(self, skill_name: str | None = None) -> None:
        self.settings = get_settings()
        self.runner = AgentRunner(self.settings)
        self.lock = threading.Lock()
        if skill_name:
            self.runner.set_skill(skill_name)

    def ask(self, prompt: str) -> str:
        with self.lock:
            return self.runner.ask(prompt)

    def ask_with_reasoning(self, prompt: str) -> dict[str, str]:
        with self.lock:
            return self.runner.ask_with_reasoning(prompt)

    def reset(self) -> None:
        with self.lock:
            self.runner.reset_history()

    def list_skills(self) -> list[str]:
        with self.lock:
            return self.runner.list_skills()

    def skill_options(self) -> list[dict[str, Any]]:
        with self.lock:
            return self.runner.skill_options()

    def set_skill(self, skill_name: str | None) -> str:
        with self.lock:
            return self.runner.set_skill(skill_name)

    def active_skill(self) -> str:
        with self.lock:
            if self.runner.active_skill is None:
                return "none"
            return self.runner.active_skill.name

    def fast_reply(self) -> bool:
        with self.lock:
            return self.runner.fast_reply

    def set_fast_reply(self, enabled: bool) -> str:
        with self.lock:
            return self.runner.set_fast_reply(enabled)

    def summarize_document(self, filename: str, content_base64: str) -> dict[str, str]:
        raw = base64.b64decode(content_base64)
        upload_path = _save_upload(filename, raw)
        document_text = extract_document_text(upload_path, raw)
        if not document_text.strip():
            return {
                "reply": f"I uploaded {upload_path.name}, but could not extract readable text from it.",
                "reasoning": "",
                "path": str(upload_path.relative_to(PROJECT_ROOT)),
            }

        prompt = (
            "Summarize only the document text below. Do not use prior chat history or invent "
            "details. Include main points and action items. If the document is short, say so.\n\n"
            f"Filename: {upload_path.name}\n\n"
            f"{document_text[:12000]}"
        )
        reply = self.runner.ask_once(prompt, max_tokens=320)
        return {
            "reply": reply,
            "reasoning": "",
            "path": str(upload_path.relative_to(PROJECT_ROOT)),
        }

    def generate_image(self, prompt: str) -> dict[str, str]:
        try:
            image_path = generate_openai_image(prompt)
            note = "Generated image with the configured OpenAI image model."
        except Exception as exc:
            image_path = generate_svg_image(prompt)
            note = (
                "Created a local image prompt card. Add a real OpenAI API key and "
                "OPENAI_IMAGE_MODEL in .env for true AI image generation. "
                f"Fallback reason: {exc.__class__.__name__}: {exc}"
            )
        return {
            "reply": f"{note}\nSaved: {image_path.relative_to(PROJECT_ROOT)}",
            "reasoning": "",
            "url": "/" + image_path.relative_to(PROJECT_ROOT).as_posix(),
        }

    def generate_video(self, prompt: str) -> dict[str, str]:
        path = generate_video_storyboard(prompt)
        return {
            "reply": (
                "Created a local animated video storyboard from your prompt. "
                "This app does not have a configured video-generation provider yet.\n"
                f"Saved: {path.relative_to(PROJECT_ROOT)}"
            ),
            "reasoning": "",
            "url": "/" + path.relative_to(PROJECT_ROOT).as_posix(),
        }

    def history(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        with self.lock:
            for item in self.runner.history:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
        return messages

    def export_chat_markdown(self) -> str:
        with self.lock:
            lines = ["# Neo Conversation Export\n"]
            for item in self.runner.history:
                role = item.get("role")
                content = item.get("content")
                if role == "user" and isinstance(content, str):
                    lines.append(f"### User\n\n{content}\n")
                elif role == "assistant" and isinstance(content, str):
                    lines.append(f"### Neo\n\n{content}\n")
                elif role == "tool" and isinstance(content, str):
                    lines.append(f"#### Tool Output\n\n```\n{content}\n```\n")
            return "\n".join(lines)

    def transcribe_audio(self, filename: str, content_base64: str) -> dict[str, str]:
        raw = base64.b64decode(content_base64)
        adapter = WhisperSTTAdapter(self.settings.api_key)
        transcript = adapter.transcribe(raw, filename)
        return {"transcript": transcript}

    def normalize_speech_text(self, text: str) -> dict[str, str]:
        normalized = normalize_for_tts(text)
        return {"normalized": normalized}


def _safe_filename(filename: str, default_suffix: str = ".txt") -> str:
    name = Path(filename or "upload").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        name = "upload"
    if "." not in name and default_suffix:
        name += default_suffix
    return name


def _save_upload(filename: str, raw: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    path.write_bytes(raw)
    return path


def extract_document_text(path: Path, raw: bytes) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".css", ".xml"}:
        return raw.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = raw.decode("utf-8", errors="replace")
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    return raw.decode("utf-8", errors="replace")


def _extract_docx_text(path: Path) -> str:
    import xml.etree.ElementTree as ET

    with ZipFile(path) as docx:
        xml_bytes = docx.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            paragraphs.append(node.text)
        elif node.tag.endswith("}p"):
            paragraphs.append("\n")
    return " ".join(paragraphs).replace(" \n ", "\n").strip()


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdf. Run: .\\.venv\\Scripts\\python.exe -m pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages[:40]]
    return "\n\n".join(pages).strip()


def generate_openai_image(prompt: str) -> Path:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_IMAGE_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key or api_key == "ollama":
        raise RuntimeError("No real OpenAI image API key is configured.")

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
    size = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024").strip() or "1024x1024"
    client = OpenAI(api_key=api_key)
    response = client.images.generate(model=model, prompt=prompt, size=size)
    image = response.data[0]
    if not getattr(image, "b64_json", None):
        raise RuntimeError("Image API did not return base64 image data.")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"image_{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(base64.b64decode(image.b64_json))
    return path


def generate_svg_image(prompt: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    title = html.escape(prompt[:90] or "Generated image")
    wrapped = _wrap_svg_text(prompt, 38, 7)
    lines = "\n".join(
        f'<text x="64" y="{250 + index * 36}" class="prompt">{html.escape(line)}</text>'
        for index, line in enumerate(wrapped)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#164e63"/>
      <stop offset="0.55" stop-color="#f8fafc"/>
      <stop offset="1" stop-color="#b45309"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#g)"/>
  <rect x="48" y="48" width="928" height="928" rx="28" fill="rgba(255,255,255,0.84)"/>
  <text x="64" y="132" font-family="Segoe UI, Arial" font-size="42" font-weight="700" fill="#111827">{title}</text>
  <text x="64" y="190" font-family="Segoe UI, Arial" font-size="22" fill="#475569">Local image artifact</text>
  <style>.prompt {{ font: 30px Segoe UI, Arial; fill: #18202b; }}</style>
  {lines}
</svg>
"""
    path = GENERATED_DIR / f"image_prompt_{uuid.uuid4().hex[:8]}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def generate_video_storyboard(prompt: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    safe_prompt = html.escape(prompt)
    shots = _wrap_svg_text(prompt, 42, 5) or ["Opening shot", "Main action", "Closing shot"]
    shot_cards = "\n".join(
        f'<section><strong>Shot {index + 1}</strong><p>{html.escape(shot)}</p></section>'
        for index, shot in enumerate(shots)
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Storyboard</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #111827; color: #f8fafc; }}
    main {{ min-height: 100vh; display: grid; place-items: center; padding: 32px; }}
    .frame {{ width: min(960px, 100%); aspect-ratio: 16 / 9; border: 1px solid #334155; overflow: hidden; position: relative; background: linear-gradient(135deg, #0f766e, #f8fafc 55%, #a16207); }}
    .caption {{ position: absolute; left: 32px; right: 32px; bottom: 30px; padding: 18px; background: rgba(17,24,39,.78); border-radius: 8px; font-size: clamp(18px, 3vw, 34px); }}
    .orb {{ position: absolute; width: 180px; height: 180px; background: #f8fafc; opacity: .65; animation: move 7s infinite alternate ease-in-out; }}
    @keyframes move {{ from {{ transform: translate(80px, 90px) rotate(0deg); }} to {{ transform: translate(660px, 220px) rotate(38deg); }} }}
    .shots {{ width: min(960px, 100%); display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 14px; }}
    section {{ border: 1px solid #334155; border-radius: 8px; padding: 12px; background: #1f2937; }}
  </style>
</head>
<body>
  <main>
    <div>
      <div class="frame"><div class="orb"></div><div class="caption">{safe_prompt}</div></div>
      <div class="shots">{shot_cards}</div>
    </div>
  </main>
</body>
</html>
"""
    path = GENERATED_DIR / f"video_storyboard_{uuid.uuid4().hex[:8]}.html"
    path.write_text(page, encoding="utf-8")
    return path


def _wrap_svg_text(text: str, width: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:width]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def run_web_app(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    skill: str | None = None,
) -> None:
    try:
        agent = WebAgent(skill_name=skill)
    except ValueError as exc:
        print(f"Could not start web app: {exc}")
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(INDEX_HTML)
                return
            if path == "/api/history":
                self._send_json({"messages": agent.history()})
                return
            if path == "/api/export":
                md = agent.export_chat_markdown()
                body = md.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=neo_chat_export.md")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/skills":
                self._send_json({"active": agent.active_skill(), "skills": agent.skill_options()})
                return
            if path == "/api/fast":
                self._send_json({"enabled": agent.fast_reply()})
                return
            if path.startswith("/generated/") or path.startswith("/uploads/"):
                self._send_file(path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/chat":
                payload = self._read_json()
                prompt = str(payload.get("message", "")).strip()
                if not prompt:
                    self._send_json({"error": "Message is required."}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = agent.ask_with_reasoning(prompt)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send_json(result)
                return
            if path == "/api/speech-to-text":
                payload = self._read_json()
                filename = str(payload.get("filename", "audio.wav")).strip()
                content = str(payload.get("content_base64", "")).strip()
                if not content:
                    self._send_json({"error": "Audio content is required."}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = agent.transcribe_audio(filename, content)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send_json(result)
                return
            if path == "/api/text-to-speech":
                payload = self._read_json()
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._send_json({"error": "Text is required."}, HTTPStatus.BAD_REQUEST)
                    return
                result = agent.normalize_speech_text(text)
                self._send_json(result)
                return
            if path == "/api/generate-image":
                payload = self._read_json()
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    self._send_json({"error": "Prompt is required."}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(agent.generate_image(prompt))
                return
            if path == "/api/generate-video":
                payload = self._read_json()
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    self._send_json({"error": "Prompt is required."}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(agent.generate_video(prompt))
                return
            if path == "/api/document":
                payload = self._read_json()
                filename = str(payload.get("filename", "upload.txt")).strip()
                content = str(payload.get("content_base64", "")).strip()
                if not content:
                    self._send_json({"error": "Document content is required."}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = agent.summarize_document(filename, content)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send_json(result)
                return
            if path == "/api/reset":
                agent.reset()
                self._send_json({"ok": True})
                return
            if path == "/api/skill":
                payload = self._read_json()
                skill_name = str(payload.get("skill", "")).strip()
                try:
                    if not skill_name or skill_name.lower() in {"off", "none", "disable"}:
                        message = agent.set_skill(None)
                    else:
                        message = agent.set_skill(skill_name)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({"ok": True, "message": message, "active": agent.active_skill()})
                return
            if path == "/api/fast":
                payload = self._read_json()
                raw_enabled = payload.get("enabled", None)
                if raw_enabled is None:
                    self._send_json({"error": "enabled is required."}, HTTPStatus.BAD_REQUEST)
                    return
                enabled = bool(raw_enabled)
                message = agent.set_fast_reply(enabled)
                self._send_json({"ok": True, "message": message, "enabled": agent.fast_reply()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: str) -> None:
            target = (PROJECT_ROOT / path.lstrip("/")).resolve()
            if PROJECT_ROOT not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix in {".html", ".svg"}:
                content_type += "; charset=utf-8"
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    print(f"Neo web app running at {url}")
    print(f"Active skill: {agent.active_skill()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Neo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #06090d;
      --panel: rgba(9, 20, 27, 0.82);
      --panel-strong: rgba(13, 35, 45, 0.92);
      --panel-hover: rgba(18, 54, 68, 0.88);
      --text: #eafcff;
      --muted: #83aeb8;
      --line: rgba(69, 220, 241, 0.32);
      --line-strong: rgba(89, 232, 255, 0.68);
      --cyan: #44e7ff;
      --cyan-soft: rgba(68, 231, 255, 0.15);
      --gold: #f6c65b;
      --red: #ff5b4f;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.38);
    }
    body.theme-matrix {
      --bg: #030a05;
      --cyan: #22c55e;
      --gold: #15803d;
      --text: #e2fbe8;
      --muted: #86efac;
      --line: rgba(34, 197, 94, 0.32);
      --line-strong: rgba(34, 197, 94, 0.68);
    }
    body.theme-obsidian {
      --bg: #0d0a06;
      --cyan: #f59e0b;
      --gold: #fbbf24;
      --text: #fffbeb;
      --muted: #fde047;
      --line: rgba(245, 158, 11, 0.32);
      --line-strong: rgba(245, 158, 11, 0.68);
    }
    body.theme-light {
      color-scheme: light;
      --bg: #f8fafc;
      --panel: rgba(255, 255, 255, 0.95);
      --panel-strong: rgba(241, 245, 249, 0.98);
      --panel-hover: rgba(226, 232, 240, 0.9);
      --text: #0f172a;
      --muted: #475569;
      --line: rgba(2, 132, 199, 0.25);
      --line-strong: rgba(2, 132, 199, 0.55);
      --cyan: #0284c7;
      --cyan-soft: rgba(2, 132, 199, 0.1);
      --gold: #d97706;
      --red: #dc2626;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }
    .typing-indicator {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 14px;
      border-left: 2px solid var(--cyan);
      background: var(--panel);
      color: var(--muted);
      font-style: italic;
      border-radius: 4px;
    }
    .copy-btn {
      margin-top: 8px;
      padding: 4px 10px;
      font-size: 11px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--panel-strong);
      color: var(--cyan);
      cursor: pointer;
    }
    .copy-btn:hover {
      background: var(--panel-hover);
    }
    .typing-dot {
      width: 6px;
      height: 6px;
      background: var(--cyan);
      border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1.0); }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(rgba(68, 231, 255, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(68, 231, 255, 0.045) 1px, transparent 1px),
        radial-gradient(circle at 50% 18%, rgba(68, 231, 255, 0.2), transparent 34%),
        linear-gradient(135deg, #06090d 0%, #0b171c 45%, #120b08 100%);
      background-size: 48px 48px, 48px 48px, auto, auto;
      color: var(--text);
      font-family: "Segoe UI", "Bahnschrift", Arial, sans-serif;
      letter-spacing: 0;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(rgba(255,255,255,0.035), rgba(255,255,255,0) 2px);
      background-size: 100% 4px;
      opacity: 0.28;
      z-index: 0;
    }
    .app {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 86px 1fr;
    }
    .rail {
      background: linear-gradient(180deg, rgba(7, 22, 30, 0.94), rgba(8, 11, 15, 0.88));
      border-right: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      padding: 22px 12px;
      box-shadow: inset -1px 0 0 rgba(246, 198, 91, 0.18);
    }
    .rail-mark {
      width: 52px;
      height: 52px;
      border: 0;
      border-radius: 50%;
      display: grid;
      place-items: center;
      color: var(--cyan);
      font-weight: 750;
      background:
        radial-gradient(circle, rgba(68, 231, 255, 0.22) 0 32%, transparent 33%),
        conic-gradient(from 20deg, var(--cyan), transparent 20%, var(--gold), transparent 48%, var(--cyan));
      box-shadow: 0 0 26px rgba(68, 231, 255, 0.4), inset 0 0 18px rgba(68, 231, 255, 0.28);
    }
    .rail button {
      width: 46px;
      height: 46px;
      min-width: 46px;
      border: 0;
      border-radius: 50%;
      background: rgba(68, 231, 255, 0.08);
      color: var(--cyan);
      box-shadow: inset 0 0 0 1px var(--line);
    }
    .rail button:hover { background: rgba(68, 231, 255, 0.17); box-shadow: 0 0 18px rgba(68, 231, 255, 0.28), inset 0 0 0 1px var(--line-strong); }
    .content {
      min-width: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 22px 32px 12px;
      background: linear-gradient(180deg, rgba(5, 14, 20, 0.92), rgba(5, 14, 20, 0));
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      color: var(--cyan);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      text-shadow: 0 0 18px rgba(68, 231, 255, 0.52);
    }
    .status {
      color: var(--gold);
      font-size: 12px;
      min-height: 18px;
      text-align: right;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .topbar-right {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .skill-panel {
      display: grid;
      gap: 6px;
      min-width: min(360px, 100%);
      justify-items: end;
    }
    .skill-panel label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    .skill-row {
      display: flex;
      align-items: center;
      gap: 8px;
      width: min(360px, 100%);
    }
    .skill-row select {
      flex: 1;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(9, 20, 27, 0.96);
      color: var(--text);
      padding: 10px 14px;
      font: inherit;
      box-shadow: inset 0 0 0 1px rgba(68, 231, 255, 0.04);
    }
    .skill-row select:focus {
      outline: none;
      border-color: var(--line-strong);
      box-shadow: 0 0 0 3px rgba(68, 231, 255, 0.12);
    }
    .skill-row button {
      white-space: nowrap;
      min-width: 88px;
    }
    .fast-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(9, 20, 27, 0.92);
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    .fast-toggle.active {
      border-color: rgba(246, 198, 91, 0.7);
      box-shadow: 0 0 0 3px rgba(246, 198, 91, 0.12);
    }
    .fast-toggle span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    main {
      width: min(1080px, 100%);
      margin: 0 auto;
      padding: 18px 24px 30px;
      overflow: auto;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 34px;
      align-items: center;
      padding: 42px 0 30px;
    }
    .hero-title {
      margin: 0;
      font-size: clamp(42px, 7vw, 76px);
      line-height: 1.08;
      font-weight: 750;
      letter-spacing: 0;
      color: var(--text);
      text-transform: uppercase;
    }
    .hero-title span {
      background: linear-gradient(90deg, var(--cyan), #ffffff, var(--gold));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      text-shadow: none;
    }
    .hero-subtitle {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: clamp(20px, 3vw, 32px);
      font-weight: 500;
    }
    .hero.hidden { display: none; }
    .reactor {
      width: 246px;
      aspect-ratio: 1;
      justify-self: end;
      border-radius: 50%;
      position: relative;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle, rgba(234,252,255,0.9) 0 9%, rgba(68,231,255,0.38) 10% 18%, transparent 19%),
        repeating-conic-gradient(from 0deg, rgba(68,231,255,0.82) 0deg 8deg, transparent 8deg 17deg),
        radial-gradient(circle, transparent 0 45%, rgba(68,231,255,0.28) 46% 48%, transparent 49% 62%, rgba(246,198,91,0.18) 63% 65%, transparent 66%);
      box-shadow: 0 0 46px rgba(68, 231, 255, 0.38), inset 0 0 34px rgba(68, 231, 255, 0.28);
    }
    .reactor::before,
    .reactor::after {
      content: "";
      position: absolute;
      inset: 22px;
      border-radius: 50%;
      border: 1px solid var(--line-strong);
    }
    .reactor::after {
      inset: 62px;
      border-color: rgba(246, 198, 91, 0.6);
      box-shadow: 0 0 22px rgba(246, 198, 91, 0.22);
    }
    .reactor-label {
      position: relative;
      z-index: 1;
      color: var(--cyan);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-shadow: 0 0 14px rgba(68, 231, 255, 0.9);
    }
    .prompt-cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 42px;
      grid-column: 1 / -1;
    }
    .prompt-card {
      min-height: 138px;
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: linear-gradient(145deg, rgba(8, 28, 38, 0.86), rgba(20, 19, 15, 0.78));
      color: var(--text);
      text-align: left;
      align-content: space-between;
      line-height: 1.35;
      box-shadow: inset 0 0 24px rgba(68, 231, 255, 0.06);
    }
    .prompt-card:hover {
      background: var(--panel-hover);
      border-color: var(--line-strong);
      box-shadow: 0 0 22px rgba(68, 231, 255, 0.18), inset 0 0 28px rgba(68, 231, 255, 0.08);
    }
    .prompt-card svg {
      justify-self: end;
      color: var(--gold);
    }
    .messages {
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .message {
      max-width: min(780px, 94%);
      line-height: 1.58;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
    }
    .message.user {
      align-self: flex-end;
      padding: 13px 18px;
      border: 1px solid rgba(246, 198, 91, 0.42);
      border-radius: 6px 6px 2px 6px;
      background: rgba(70, 43, 17, 0.58);
      color: #fff2cf;
      box-shadow: inset 0 0 18px rgba(246, 198, 91, 0.08);
    }
    .message.assistant {
      align-self: flex-start;
      padding: 14px 16px;
      border-left: 2px solid var(--cyan);
      background: linear-gradient(90deg, rgba(68, 231, 255, 0.12), rgba(68, 231, 255, 0.02));
      color: var(--text);
      box-shadow: -12px 0 24px rgba(68, 231, 255, 0.08);
    }
    .reasoning {
      max-width: min(780px, 94%);
      align-self: flex-start;
      padding: 12px 14px;
      border: 1px solid rgba(246, 198, 91, 0.42);
      border-radius: 6px;
      background: rgba(39, 29, 13, 0.62);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .reasoning-title {
      display: block;
      margin-bottom: 4px;
      color: var(--gold);
      font-weight: 650;
    }
    .composer {
      background: linear-gradient(180deg, rgba(6, 9, 13, 0), rgba(6, 9, 13, 0.96) 30%);
      padding: 18px 24px 24px;
    }
    .composer-inner {
      width: min(1080px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr;
      gap: 4px;
      align-items: end;
      padding: 12px 12px 10px;
      border: 1px solid var(--line-strong);
      border-radius: 10px;
      background: linear-gradient(180deg, rgba(10, 35, 45, 0.94), rgba(8, 18, 24, 0.94));
      box-shadow: var(--shadow);
    }
    textarea {
      width: 100%;
      min-height: 44px;
      max-height: 150px;
      resize: vertical;
      padding: 12px 12px 10px 16px;
      border: 0;
      outline: 0;
      border-radius: 6px;
      background: rgba(3, 12, 17, 0.46);
      color: var(--text);
      font: inherit;
      line-height: 1.35;
    }
    textarea::placeholder { color: rgba(131, 174, 184, 0.86); }
    .actions {
      display: flex;
      align-items: center;
      gap: 6px;
      justify-content: space-between;
    }
    .actions-left,
    .actions-right {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    button {
      min-width: 42px;
      height: 42px;
      border: 1px solid rgba(68, 231, 255, 0.2);
      border-radius: 6px;
      background: rgba(68, 231, 255, 0.08);
      color: var(--cyan);
      font: inherit;
      cursor: pointer;
      display: inline-grid;
      place-items: center;
      flex: 0 0 auto;
      position: relative;
    }
    button:hover { background: rgba(68, 231, 255, 0.18); border-color: var(--line-strong); box-shadow: 0 0 16px rgba(68, 231, 255, 0.18); }
    button.primary {
      min-width: 42px;
      background: linear-gradient(135deg, rgba(68, 231, 255, 0.92), rgba(246, 198, 91, 0.85));
      color: #061116;
      font-weight: 700;
    }
    button.primary:hover { background: linear-gradient(135deg, #8cf3ff, #ffd77a); }
    button.active {
      color: var(--gold);
      background: rgba(246, 198, 91, 0.13);
      border-color: rgba(246, 198, 91, 0.55);
    }
    button.danger {
      color: var(--red);
    }
    svg.icon {
      width: 20px;
      height: 20px;
      stroke: currentColor;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
      fill: none;
      pointer-events: none;
    }
    .button-label {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }
    .artifact-link {
      display: inline-block;
      margin-top: 8px;
      color: var(--gold);
      font-weight: 600;
      text-decoration: none;
    }
    .artifact-link:hover { text-decoration: underline; }
    .hint {
      width: min(920px, 100%);
      margin: 10px auto 0;
      color: rgba(131, 174, 184, 0.82);
      font-size: 12px;
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    @media (max-width: 640px) {
      .app { grid-template-columns: 1fr; }
      .rail { display: none; }
      .topbar { align-items: flex-start; flex-direction: column; padding: 16px 18px 6px; }
      .topbar-right { width: 100%; justify-content: flex-start; }
      .skill-panel { width: 100%; justify-items: start; }
      .skill-row { width: 100%; }
      .fast-toggle { width: 100%; justify-content: space-between; }
      .status { text-align: left; }
      main { padding: 10px 16px 22px; }
      .hero { grid-template-columns: 1fr; padding-top: 34px; }
      .reactor { width: 190px; justify-self: center; order: -1; }
      .prompt-cards { grid-template-columns: 1fr 1fr; margin-top: 28px; }
      .prompt-card { min-height: 112px; padding: 12px; }
      .composer { padding: 12px 12px 16px; }
      .composer-inner { grid-template-columns: 1fr; border-radius: 10px; }
      textarea { min-height: 42px; }
    }
    @media (max-width: 420px) {
      .prompt-cards { grid-template-columns: 1fr; }
      .actions { gap: 4px; }
      button { min-width: 38px; height: 38px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="rail" aria-label="App navigation">
      <div class="rail-mark" title="Neo">N</div>
      <button id="newChat" title="New chat" type="button" aria-label="New chat">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
      </button>
      <button id="exportChat" title="Export conversation as Markdown" type="button" aria-label="Export chat">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
    </aside>
    <div class="content">
      <header class="topbar">
        <h1>Neo</h1>
        <div class="topbar-right">
          <div class="skill-panel" aria-label="Theme selection">
            <label for="themeSelect">Theme</label>
            <div class="skill-row">
              <select id="themeSelect" aria-label="Choose visual theme">
                <option value="cyberpunk">Cyberpunk Cyan</option>
                <option value="matrix">Matrix Green</option>
                <option value="obsidian">Obsidian Gold</option>
                <option value="light">Clean Light</option>
              </select>
            </div>
          </div>
          <div class="skill-panel" aria-label="Skill selection">
            <label for="skillSelect">Skill mode</label>
            <div class="skill-row">
              <select id="skillSelect" aria-label="Choose an active skill">
                <option value="">Loading skills...</option>
              </select>
              <button id="skillApply" type="button" title="Apply selected skill">Apply</button>
            </div>
          </div>
          <button id="fastToggle" class="fast-toggle" type="button" aria-pressed="false" title="Toggle fast reply mode">
            <span>Fast reply</span>
            <strong id="fastToggleState">Off</strong>
          </button>
          <div id="status" class="status">Systems ready</div>
        </div>
      </header>
      <main>
        <section id="hero" class="hero">
          <div>
            <h2 class="hero-title"><span>Neo</span></h2>
            <p class="hero-subtitle">Online and standing by.</p>
          </div>
          <div class="reactor" aria-hidden="true"><span class="reactor-label">ONLINE</span></div>
          <div class="prompt-cards" aria-label="Prompt suggestions">
            <button class="prompt-card" type="button" data-prompt="Run a quick systems-style briefing for my current task">
              <span>Run a quick systems briefing for my current task</span>
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h10M4 17h7"/></svg>
            </button>
            <button class="prompt-card" type="button" data-prompt="Analyze this plan and identify the highest risk items">
              <span>Analyze this plan and identify high-risk items</span>
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6h1M4 12h1M4 18h1"/></svg>
            </button>
            <button class="prompt-card" type="button" data-prompt="Create a futuristic HUD image prompt">
              <span>Create a futuristic HUD image prompt</span>
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m8 13 2.5-2.5L15 15l2-2 3 3"/><circle cx="8" cy="9" r="1"/></svg>
            </button>
            <button class="prompt-card" type="button" data-prompt="Draft a precise response in a calm technical tone">
              <span>Draft a precise response in a calm technical tone</span>
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>
            </button>
          </div>
        </section>
        <div id="messages" class="messages"></div>
      </main>
      <section class="composer">
        <div class="composer-inner">
          <textarea id="input" placeholder="Enter a command"></textarea>
          <div class="actions">
            <div class="actions-left">
              <button id="upload" title="Upload and summarize a document" type="button" aria-label="Upload document">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
              </button>
              <button id="image" title="Generate image from the prompt" type="button" aria-label="Generate image">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m8 13 2.5-2.5L15 15l2-2 3 3"/><circle cx="8" cy="9" r="1"/></svg>
              </button>
              <button id="video" title="Generate video storyboard from the prompt" type="button" aria-label="Generate storyboard">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9 5 3-5 3z"/></svg>
              </button>
              <button id="voiceRecord" title="Record Telugu voice message" type="button" aria-label="Record voice">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
              </button>
              <button id="stopVoiceAudio" title="Barge-in / Stop speaking" class="danger" type="button" aria-label="Stop speech">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/><rect x="9" y="9" width="6" height="6"/></svg>
              </button>
            </div>
            <div class="actions-right">
              <button id="speak" title="Toggle spoken replies" class="active" type="button" aria-label="Toggle spoken replies">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15 9.5a4 4 0 0 1 0 5"/><path d="M18 7a8 8 0 0 1 0 10"/></svg>
              </button>
              <button id="mic" title="Speak" type="button" aria-label="Speak">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/></svg>
              </button>
              <button id="send" class="primary" title="Send" type="button" aria-label="Send">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>
              </button>
            </div>
          </div>
        </div>
        <input id="file" type="file" hidden accept=".txt,.md,.csv,.json,.html,.htm,.docx,.pdf">
        <div class="hint">Local memory active | Voice channel armed | Tool modules online</div>
      </section>
    </div>
  </div>
  <script>
    const messages = document.getElementById("messages");
    const hero = document.getElementById("hero");
    const promptCards = document.querySelectorAll(".prompt-card");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const newChat = document.getElementById("newChat");
    const mic = document.getElementById("mic");
    const upload = document.getElementById("upload");
    const fileInput = document.getElementById("file");
    const image = document.getElementById("image");
    const video = document.getElementById("video");
    const speakToggle = document.getElementById("speak");
    const skillSelect = document.getElementById("skillSelect");
    const skillApply = document.getElementById("skillApply");
    const fastToggle = document.getElementById("fastToggle");
    const fastToggleState = document.getElementById("fastToggleState");
    const status = document.getElementById("status");
    let spokenReplies = true;
    let recognition = null;

    function updateHero() {
      hero.classList.toggle("hidden", messages.children.length > 0);
    }

    function renderMarkdown(text) {
      const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      return escaped
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    }

    let typingElement = null;
    function addTypingIndicator() {
      if (typingElement) return;
      const div = document.createElement("div");
      div.className = "typing-indicator";
      div.innerHTML = 'Neo is thinking <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
      messages.appendChild(div);
      updateHero();
      div.scrollIntoView({ behavior: "smooth", block: "end" });
      typingElement = div;
    }

    function removeTypingIndicator() {
      if (typingElement) {
        typingElement.remove();
        typingElement = null;
        updateHero();
      }
    }

    function addMessage(role, content) {
      const div = document.createElement("div");
      div.className = `message ${role}`;
      const contentDiv = document.createElement("div");
      contentDiv.innerHTML = renderMarkdown(content);
      div.appendChild(contentDiv);

      if (role === "assistant") {
        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-btn";
        copyBtn.textContent = "Copy";
        copyBtn.title = "Copy message text";
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(content);
          copyBtn.textContent = "Copied!";
          setTimeout(() => copyBtn.textContent = "Copy", 1500);
        };
        div.appendChild(copyBtn);
      }

      messages.appendChild(div);
      updateHero();
      div.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function addArtifact(role, content, url, label) {
      const div = document.createElement("div");
      div.className = `message ${role}`;
      const text = document.createElement("div");
      text.innerHTML = renderMarkdown(content);
      div.appendChild(text);
      if (url) {
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.className = "artifact-link";
        link.textContent = label || "Open file";
        div.appendChild(link);
      }
      messages.appendChild(div);
      updateHero();
      div.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function addReasoning(content) {
      if (!content || !content.trim()) return;
      const div = document.createElement("div");
      div.className = "reasoning";
      const title = document.createElement("span");
      title.className = "reasoning-title";
      title.textContent = "Reasoning summary";
      div.appendChild(title);
      div.appendChild(document.createTextNode(content));
      messages.appendChild(div);
      updateHero();
      div.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function setStatus(text) {
      status.textContent = text;
    }

    function hasTelugu(text) {
      return /[\u0c00-\u0c7f]/.test(text);
    }

    async function speak(text) {
      if (!spokenReplies || !("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      try {
        const res = await fetch("/api/text-to-speech", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        const data = await res.json();
        const spokenText = data.normalized || text;
        const utterance = new SpeechSynthesisUtterance(spokenText);
        const isTelugu = hasTelugu(spokenText) || (skillSelect && skillSelect.value === "telugu");
        utterance.lang = isTelugu ? "te-IN" : "en-US";
        utterance.rate = 1;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        const utterance = new SpeechSynthesisUtterance(text);
        const isTelugu = hasTelugu(text) || (skillSelect && skillSelect.value === "telugu");
        utterance.lang = isTelugu ? "te-IN" : "en-US";
        window.speechSynthesis.speak(utterance);
      }
    }

    async function sendMessage(text) {
      const prompt = text.trim();
      if (!prompt) return;
      input.value = "";
      addMessage("user", prompt);
      setStatus("Thinking...");
      addTypingIndicator();
      send.disabled = true;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: prompt })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");
        removeTypingIndicator();
        addReasoning(data.reasoning);
        addMessage("assistant", data.reply);
        speak(data.reply);
        setStatus("Ready");
      } catch (error) {
        removeTypingIndicator();
        addMessage("assistant", `Error: ${error.message}`);
        setStatus("Error");
      } finally {
        removeTypingIndicator();
        send.disabled = false;
        input.focus();
      }
    }

    async function postSkill(endpoint, payload, statusText, linkLabel) {
      setStatus(statusText);
      send.disabled = true;
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");
        addReasoning(data.reasoning);
        addArtifact("assistant", data.reply, data.url, linkLabel);
        speak(data.reply);
        setStatus("Ready");
      } catch (error) {
        addMessage("assistant", `Error: ${error.message}`);
        setStatus("Error");
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    function setSkillStatus(message) {
      setStatus(message || "Ready");
    }

    function renderSkillOptions(skills, active) {
      skillSelect.replaceChildren();
      const offOption = document.createElement("option");
      offOption.value = "";
      offOption.textContent = "No skill";
      skillSelect.appendChild(offOption);

      for (const skill of skills) {
        const option = document.createElement("option");
        option.value = skill.name;
        option.textContent = skill.description ? `${skill.name} - ${skill.description}` : skill.name;
        option.dataset.description = skill.description || "";
        skillSelect.appendChild(option);
      }

      skillSelect.value = active && active !== "none" ? active : "";
    }

    function renderFastState(enabled) {
      fastToggle.classList.toggle("active", enabled);
      fastToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
      fastToggleState.textContent = enabled ? "On" : "Off";
    }

    async function loadSkills() {
      try {
        const response = await fetch("/api/skills");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not load skills");
        renderSkillOptions(data.skills || [], data.active || "");
        const activeSkill = data.active && data.active !== "none" ? data.active : "none";
        setSkillStatus(`Skill: ${activeSkill}`);
      } catch (error) {
        skillSelect.replaceChildren();
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Skills unavailable";
        skillSelect.appendChild(option);
        skillSelect.disabled = true;
        skillApply.disabled = true;
        setSkillStatus(`Skill load error: ${error.message}`);
      }
    }

    async function loadFastReply() {
      try {
        const response = await fetch("/api/fast");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not load fast mode");
        renderFastState(Boolean(data.enabled));
      } catch (error) {
        fastToggle.disabled = true;
        setStatus(`Fast mode error: ${error.message}`);
      }
    }

    async function applySkill(skillName) {
      const normalized = skillName || "";
      skillApply.disabled = true;
      skillSelect.disabled = true;
      try {
        const response = await fetch("/api/skill", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ skill: normalized })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not update skill");
        await loadSkills();
        if (recognition) {
          recognition.lang = normalized === "telugu" ? "te-IN" : "en-US";
        }
        setSkillStatus(data.message || "Skill updated");
      } catch (error) {
        setSkillStatus(`Skill error: ${error.message}`);
      } finally {
        skillApply.disabled = false;
        skillSelect.disabled = false;
      }
    }

    async function toggleFastReply() {
      const enabled = !fastToggle.classList.contains("active");
      fastToggle.disabled = true;
      try {
        const response = await fetch("/api/fast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not update fast mode");
        renderFastState(Boolean(data.enabled));
        setSkillStatus(data.message || "Fast mode updated");
      } catch (error) {
        setSkillStatus(`Fast mode error: ${error.message}`);
      } finally {
        fastToggle.disabled = false;
      }
    }

    function generateImage() {
      const prompt = input.value.trim();
      if (!prompt) return;
      addMessage("user", `Generate image: ${prompt}`);
      input.value = "";
      postSkill("/api/generate-image", { prompt }, "Generating image...", "Open image");
    }

    function generateVideo() {
      const prompt = input.value.trim();
      if (!prompt) return;
      addMessage("user", `Generate video: ${prompt}`);
      input.value = "";
      postSkill("/api/generate-video", { prompt }, "Generating video storyboard...", "Open storyboard");
    }

    function uploadDocument(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const bytes = new Uint8Array(reader.result);
        let binary = "";
        for (let index = 0; index < bytes.length; index += 1) {
          binary += String.fromCharCode(bytes[index]);
        }
        addMessage("user", `Summarize document: ${file.name}`);
        postSkill(
          "/api/document",
          { filename: file.name, content_base64: btoa(binary) },
          "Reading document...",
          "Uploaded file"
        );
      };
      reader.onerror = () => {
        addMessage("assistant", "Error: Could not read the selected file.");
      };
      reader.readAsArrayBuffer(file);
    }

    function setupMic() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        mic.disabled = true;
        mic.title = "Browser speech recognition is not available";
        setStatus("Browser microphone speech recognition is not available");
        return;
      }
      recognition = new SpeechRecognition();
      const isTelugu = skillSelect && skillSelect.value === "telugu";
      recognition.lang = isTelugu ? "te-IN" : "en-US";
      recognition.interimResults = false;
      recognition.continuous = false;
      recognition.onstart = () => {
        mic.classList.add("active");
        setStatus(isTelugu ? "Listening in Telugu (తెలుగు)..." : "Listening...");
      };
      recognition.onend = () => {
        mic.classList.remove("active");
        if (status.textContent.startsWith("Listening")) setStatus("Ready");
      };
      recognition.onerror = event => {
        setStatus(`Mic error: ${event.error}`);
      };
      recognition.onresult = event => {
        const text = event.results[0][0].transcript;
        input.value = text;
        sendMessage(text);
      };
    }

    send.addEventListener("click", () => sendMessage(input.value));
    promptCards.forEach(card => {
      card.addEventListener("click", () => {
        input.value = card.dataset.prompt || "";
        input.focus();
      });
    });
    newChat.addEventListener("click", async () => {
      setStatus("Clearing chat...");
      try {
        const response = await fetch("/api/reset", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Reset failed");
        messages.replaceChildren();
        updateHero();
        input.value = "";
        setStatus("New chat");
      } catch (error) {
        setStatus(`Reset error: ${error.message}`);
      } finally {
        input.focus();
      }
    });
    image.addEventListener("click", generateImage);
    video.addEventListener("click", generateVideo);
    upload.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", event => {
      uploadDocument(event.target.files[0]);
      fileInput.value = "";
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(input.value);
      }
    });
    mic.addEventListener("click", () => {
      if (recognition) recognition.start();
    });
    speakToggle.addEventListener("click", () => {
      spokenReplies = !spokenReplies;
      speakToggle.classList.toggle("active", spokenReplies);
    });
    skillApply.addEventListener("click", () => applySkill(skillSelect.value));
    skillSelect.addEventListener("change", () => applySkill(skillSelect.value));
    fastToggle.addEventListener("click", toggleFastReply);

    async function loadHistory() {
      const response = await fetch("/api/history");
      const data = await response.json();
      for (const message of data.messages) {
        addMessage(message.role, message.content);
      }
    }

    const themeSelect = document.getElementById("themeSelect");
    const exportChatBtn = document.getElementById("exportChat");

    themeSelect.addEventListener("change", () => {
      const theme = themeSelect.value;
      document.body.className = theme === "cyberpunk" ? "" : `theme-${theme}`;
      localStorage.setItem("neo_theme", theme);
    });

    const savedTheme = localStorage.getItem("neo_theme") || "cyberpunk";
    themeSelect.value = savedTheme;
    document.body.className = savedTheme === "cyberpunk" ? "" : `theme-${savedTheme}`;

    exportChatBtn.addEventListener("click", () => {
      window.location.href = "/api/export";
    });

    let mediaRecorder = null;
    let audioChunks = [];
    const voiceRecordBtn = document.getElementById("voiceRecord");
    const stopVoiceAudioBtn = document.getElementById("stopVoiceAudio");

    async function toggleVoiceRecording() {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) audioChunks.push(event.data);
        };
        mediaRecorder.onstop = async () => {
          voiceRecordBtn.classList.remove("active");
          setStatus("Transcribing audio...");
          const blob = new Blob(audioChunks, { type: 'audio/webm' });
          const reader = new FileReader();
          reader.onloadend = async () => {
            const base64data = reader.result.split(',')[1];
            try {
              const res = await fetch("/api/speech-to-text", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: "recording.webm", content_base64: base64data })
              });
              const data = await res.json();
              if (data.transcript) {
                input.value = data.transcript;
                sendMessage(data.transcript);
              } else {
                setStatus("Transcription empty");
              }
            } catch (err) {
              setStatus(`STT Error: ${err.message}`);
            }
          };
          reader.readAsDataURL(blob);
          stream.getTracks().forEach(track => track.stop());
        };
        mediaRecorder.start();
        voiceRecordBtn.classList.add("active");
        setStatus("Recording voice (Telugu/EN)... Speak now");
      } catch (err) {
        setStatus(`Mic permission denied: ${err.message}`);
      }
    }

    voiceRecordBtn.addEventListener("click", toggleVoiceRecording);
    stopVoiceAudioBtn.addEventListener("click", () => {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
      setStatus("Voice stopped / Barged in");
    });

    setupMic();
    loadSkills();
    loadFastReply();
    loadHistory();
    input.focus();
  </script>
</body>
</html>
"""
