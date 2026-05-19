#!/usr/bin/env python3
"""Static file server + Gemini chat proxy (API key from .env only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))

SYSTEM_PROMPT = (
    "You are the TRUMPF Cutting Assistant expert chatbot. You answer questions "
    "exclusively about the TRUMPF Cutting Assistant product. Your knowledge covers: "
    "what the Cutting Assistant is (an AI-powered laser cutting edge optimization "
    "tool for TruLaser machines with 6kW or higher, purchased from May 2025); its "
    "two modes (AI mode with hand scanner for Baustahl Stickstoff 5–15mm, and "
    "bandwidth mode for full material range including Edelstahl and Aluminium); "
    "its benefits (objective edge quality measurement in micrometers, no programming "
    "knowledge required, compensates for skilled worker shortage, continuous "
    "improvement through field data); technical requirements (TruLaser 2D series, "
    "min 6kW, purchase from May 2025, online updates); the three research collaboration "
    "models (anonymized data sharing, weight sharing via federated fine-tuning, "
    "on-premise code collaboration); and all public press release information about "
    "the product. If a user asks about anything unrelated to the TRUMPF Cutting "
    "Assistant, respond politely that you can only answer questions about the Cutting "
    "Assistant. Respond in the same language the user writes in. Keep answers concise "
    "and professional."
)

DEFAULT_MODEL = "gemini-2.5-flash"


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def to_gemini_contents(messages: list) -> list[dict]:
    contents: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = msg.get("content", "")
        if not isinstance(text, str) or not text.strip():
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role in ("assistant", "model"):
            contents.append({"role": "model", "parts": [{"text": text}]})
    return contents


def extract_reply(data: dict) -> str | None:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p["text"] for p in parts if isinstance(p.get("text"), str)]
        reply = "".join(texts).strip()
        return reply or None
    except (KeyError, IndexError, TypeError):
        return None


load_env()
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent?key={API_KEY}"
)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("", "/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/trumpf-cutting-assistant.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/chat":
            self.send_error(404)
            return

        if not API_KEY:
            self._json(503, {"error": "missing_api_key"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = body.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            contents = to_gemini_contents(messages)
            if not contents:
                raise ValueError("messages must include at least one user message")
        except (json.JSONDecodeError, ValueError, TypeError):
            self._json(400, {"error": "invalid_request"})
            return

        payload = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": 1024},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            GEMINI_URL,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            reply = extract_reply(data)
            if not reply:
                self._json(502, {"error": "empty_response", "detail": data})
                return
            self._json(200, {"content": [{"text": reply}]})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._json(exc.code, {"error": "gemini_error", "detail": detail})
        except urllib.error.URLError:
            self._json(502, {"error": "upstream_unreachable"})

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    if API_KEY:
        print(f"Gemini API key loaded from .env (model: {MODEL})")
    else:
        print("WARNING: GEMINI_API_KEY not set — chat runs in demo mode")
    print(f"Serving http://localhost:{PORT}/")
    HTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
