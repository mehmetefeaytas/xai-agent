"""Kart, figür ve terminal sahnelerini HTML olarak üretir.

Neden HTML? Çünkü tek bir yakalama mekanizması kullanmak istiyoruz: Playwright.
Terminal tekrarı, bölüm kartları ve figür slaytları tarayıcıda oynatılıp
kaydedilince hepsi aynı çözünürlük, aynı kare hızı ve aynı görsel dile sahip
oluyor; kare kare PNG üretip birleştirmeye gerek kalmıyor.

Terminal sahnelerindeki metin UYDURMA DEĞİL: ``build/outputs/*.txt`` altındaki
gerçek komut çıktıları okunur (bkz. capture_outputs.py).
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images"
OUTPUTS = ROOT / "build" / "outputs"

BASE_CSS = """
*, *::before, *::after { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; width: 1280px; height: 720px; overflow: hidden;
  background: #0b1220;
  font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
  color: #e8eef7;
  -webkit-font-smoothing: antialiased;
}
.stage {
  position: absolute; inset: 0;
  background:
    radial-gradient(1100px 620px at 18% -12%, #1b2b47 0%, rgba(27,43,71,0) 62%),
    radial-gradient(900px 560px at 108% 112%, #17324a 0%, rgba(23,50,74,0) 60%),
    linear-gradient(160deg, #0b1220 0%, #0e1728 100%);
}
.brand {
  position: absolute; left: 44px; bottom: 26px;
  font-size: 13px; letter-spacing: .09em; text-transform: uppercase;
  color: #5c6f8c; font-weight: 600;
}
.brand b { color: #4da3ff; font-weight: 700; }
.fade-in { opacity: 0; animation: fade .55s ease forwards; }
@keyframes fade { to { opacity: 1; transform: none; } }
.rise { transform: translateY(14px); }
"""

CARD_CSS = """
.card { position: absolute; inset: 0; padding: 88px 96px; display: flex;
        flex-direction: column; justify-content: center; }
.kicker { font-size: 17px; font-weight: 700; letter-spacing: .13em;
          text-transform: uppercase; color: #4da3ff; margin-bottom: 20px; }
.title { font-size: 62px; font-weight: 800; letter-spacing: -.02em;
         line-height: 1.06; margin: 0 0 34px; color: #ffffff; }
.rule { width: 92px; height: 4px; border-radius: 3px; background: #4da3ff;
        margin-bottom: 34px; }
.lines { list-style: none; margin: 0; padding: 0; }
.lines li { position: relative; font-size: 25px; line-height: 1.5;
            color: #c3d2e6; padding-left: 34px; margin-bottom: 17px; }
.lines li::before { content: ""; position: absolute; left: 0; top: 13px;
  width: 11px; height: 11px; border-radius: 3px; background: #4da3ff;
  opacity: .85; }
"""

FIGURE_CSS = """
.fig-wrap { position: absolute; inset: 0; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 34px 44px 74px; }
.fig-frame { width: 100%; flex: 1; display: flex; align-items: center;
             justify-content: center; overflow: hidden; padding: 12px;
             border-radius: 14px; background: #ffffff;
             box-shadow: 0 26px 70px rgba(0,0,0,.55); }
/* Yakınlaşma DIŞA doğru: kadraj hafif kırpılmış başlar, tam görünümde biter.
   Böylece sahnenin en uzun süren hâli figürün TAMAMI olur — içe doğru
   yakınlaşma başlığı ve alt notu kırpıyordu. */
.fig-frame img { width: 100%; height: 100%; object-fit: contain;
                 display: block; transform-origin: var(--ox) var(--oy);
                 animation: kb var(--kbdur) ease-out var(--kbdelay) forwards; }
@keyframes kb { from { transform: scale(var(--kbscale)); } to { transform: scale(1); } }
.caption { margin-top: 20px; font-size: 21px; font-weight: 600; color: #9fb4d0;
           text-align: center; letter-spacing: .01em; }
"""

TERMINAL_CSS = """
.term-wrap { position: absolute; inset: 0; padding: 40px 46px 62px;
             display: flex; flex-direction: column; }
.term { flex: 1; display: flex; flex-direction: column; overflow: hidden;
        border-radius: 12px; background: #0a0f18;
        border: 1px solid #22304a;
        box-shadow: 0 24px 64px rgba(0,0,0,.6); }
.bar { display: flex; align-items: center; gap: 9px; padding: 12px 16px;
       background: #16202f; border-bottom: 1px solid #22304a; flex: none; }
.dot { width: 12px; height: 12px; border-radius: 50%; }
.bar .t { margin-left: 12px; font-size: 13px; color: #7d8fa8;
          font-family: Menlo, monospace; }
.body { flex: 1; overflow: hidden; padding: 16px 20px 18px;
        font-family: Menlo, "SF Mono", monospace; font-size: var(--fs);
        line-height: 1.42; white-space: pre-wrap; word-break: break-word; }
.scroller { transition: transform .22s linear; }
.prompt { color: #56d364; }
.prompt .path { color: #4da3ff; }
.prompt .cmd { color: #e8eef7; }
.cursor { display: inline-block; width: 8px; background: #e8eef7;
          animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.ln { display: block; opacity: 0; }
.ln.on { opacity: 1; }
.ok { color: #56d364; } .bad { color: #ff7b72; } .warn { color: #e3b341; }
.hl { color: #79c0ff; } .dim { color: #6e7d94; } .strong { color: #ffffff; font-weight: 700; }
.excerpt-note { flex: none; padding: 8px 20px 12px; font-size: 13px;
                color: #6e7d94; font-family: Menlo, monospace;
                border-top: 1px solid #1b2740; }
"""


# --------------------------------------------------------------------------
def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _shell(title: str, body: str, css: str, script: str = "") -> str:
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{BASE_CSS}{css}</style>
<div class="stage"></div>
{body}
<div class="brand">XAI&#8209;Agent · <b>Mehmet Efe Aytaş</b></div>
<script>{script}</script>"""


# --------------------------------------------------------------------------
# Kart
# --------------------------------------------------------------------------
def card_html(payload: dict, duration: float) -> str:
    lines = payload.get("lines") or []
    items = "\n".join(
        f'<li class="fade-in rise" style="animation-delay:{0.55 + i * 0.32:.2f}s">'
        f"{html.escape(str(t))}</li>"
        for i, t in enumerate(lines)
    )
    body = f"""<div class="card">
  <div class="kicker fade-in" style="animation-delay:.08s">{html.escape(payload.get("kicker", ""))}</div>
  <h1 class="title fade-in rise" style="animation-delay:.20s">{html.escape(payload.get("title", ""))}</h1>
  <div class="rule fade-in" style="animation-delay:.42s"></div>
  <ul class="lines">{items}</ul>
</div>"""
    return _shell(payload.get("title", "kart"), body, CARD_CSS)


# --------------------------------------------------------------------------
# Figür slaytı (yavaş yakınlaşma ile)
# --------------------------------------------------------------------------
def figure_html(payload: dict, duration: float) -> str:
    image = IMAGES / payload["image"]
    if not image.exists():
        raise FileNotFoundError(f"{image} yok — önce scripts/make_figures.py çalıştır")
    zoom = payload.get("zoom") or {}
    scale = float(zoom.get("scale", 1.05))
    ox = f"{float(zoom.get('x', 0.5)) * 100:.0f}%"
    oy = f"{float(zoom.get('y', 0.5)) * 100:.0f}%"
    # Hareket sahnenin ilk yarısında bitsin; kalan süre boyunca figür sabit ve
    # tam görünür kalır, izleyici okuyabilir.
    delay = 0.5
    kbdur = max(duration * 0.45, 1.5)

    body = f"""<div class="fig-wrap">
  <div class="fig-frame fade-in" style="animation-delay:.10s;
       --ox:{ox}; --oy:{oy}; --kbscale:{scale}; --kbdur:{kbdur:.2f}s; --kbdelay:{delay:.2f}s">
    <img src="{_data_uri(image)}" alt="">
  </div>
  <div class="caption fade-in" style="animation-delay:.45s">{html.escape(payload.get("caption", ""))}</div>
</div>"""
    return _shell(payload.get("caption", "figür"), body, FIGURE_CSS)


# --------------------------------------------------------------------------
# Terminal tekrarı
# --------------------------------------------------------------------------
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*(✅|✓).*"), "ok"),
    (re.compile(r".*(GEÇTİ|geçti|temellendirildi|BAŞARILI|passed).*"), "ok"),
    (re.compile(r"^\s*(⚠️|⚠|❌).*"), "warn"),
    (re.compile(r".*(ihlal|BAŞARISIZ|failed|error|Error|ZAYIF).*"), "bad"),
    (re.compile(r"^\s*(=|-){10,}\s*$"), "dim"),
    (re.compile(r"^\s*(KATMAN|SORU|ÖZET|AJANA GİDEN|\[tool)"), "strong"),
    (re.compile(r"^\s*[▲▼]"), "hl"),
)


def _classify(line: str) -> str:
    for pattern, cls in _RULES:
        if pattern.match(line) if pattern.pattern.startswith("^") else pattern.search(line):
            return cls
    return ""


def _load_output(payload: dict) -> tuple[list[str], bool]:
    path = OUTPUTS / f"{payload['output_key']}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok — önce scripts/video/capture_outputs.py çalıştır"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    trimmed = False

    excerpt = payload.get("excerpt")
    if excerpt:
        start_m, end_m = excerpt
        start = next((i for i, ln in enumerate(lines) if start_m in ln), 0)
        # Başlık çizgisi bir üst satırda oluyor; onu da al.
        start = max(0, start - 1)
        rest = lines[start + 1:]
        end_rel = next((i for i, ln in enumerate(rest) if end_m in ln), len(rest))
        lines = lines[start:start + 1 + end_rel]
        trimmed = True

    max_lines = payload.get("max_lines")
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        trimmed = True

    while lines and not lines[-1].strip():
        lines.pop()
    return lines, trimmed


def terminal_html(payload: dict, duration: float, command: str) -> str:
    lines, trimmed = _load_output(payload)
    n = max(len(lines), 1)
    font_size = 15 if n <= 26 else (13.5 if n <= 38 else 12)

    rendered = "".join(
        f'<span class="ln {_classify(ln)}" data-i="{i}">{html.escape(ln) or "&nbsp;"}</span>'
        for i, ln in enumerate(lines)
    )
    note = (
        '<div class="excerpt-note">↑ gerçek çalıştırma çıktısı · '
        "yer kazanmak için kısaltıldı</div>"
        if trimmed
        else ""
    )

    body = f"""<div class="term-wrap">
  <div class="term fade-in" style="animation-delay:.06s">
    <div class="bar">
      <span class="dot" style="background:#ff5f57"></span>
      <span class="dot" style="background:#febc2e"></span>
      <span class="dot" style="background:#28c840"></span>
      <span class="t">{html.escape(payload.get("title", "zsh"))}</span>
    </div>
    <div class="body" style="--fs:{font_size}px">
      <div class="scroller" id="sc"><span class="prompt">➜ <span class="path">xai-agent</span> $ <span class="cmd" id="cmd"></span><span class="cursor" id="cur">&nbsp;</span></span>
<span id="out">{rendered}</span></div>
    </div>
    {note}
  </div>
</div>"""

    script = f"""
const CMD = {json.dumps(command)};
const DUR = {duration:.3f} * 1000;
const N = {n};
const cmdEl = document.getElementById('cmd');
const curEl = document.getElementById('cur');
const sc = document.getElementById('sc');
const lns = [...document.querySelectorAll('.ln')];
const bodyEl = document.querySelector('.body');

const typeMs = Math.min(2600, Math.max(700, CMD.length * 42));
const tail = 1100;
const revealWindow = Math.max(600, DUR - typeMs - 700 - tail);
const step = Math.min(140, revealWindow / N);

let i = 0;
function typeChar() {{
  if (i <= CMD.length) {{
    cmdEl.textContent = CMD.slice(0, i);
    i++;
    setTimeout(typeChar, typeMs / Math.max(CMD.length, 1));
  }} else {{
    curEl.style.display = 'none';
    setTimeout(reveal, 340);
  }}
}}

let j = 0;
function reveal() {{
  if (j >= N) return;
  lns[j].classList.add('on');
  // son satır görünür kalsın: gerekirse yukarı kaydır
  const maxScroll = Math.max(0, sc.scrollHeight - bodyEl.clientHeight + 12);
  const upto = lns[j].offsetTop + lns[j].offsetHeight - bodyEl.clientHeight + 30;
  sc.style.transform = 'translateY(' + (-Math.max(0, Math.min(maxScroll, upto))) + 'px)';
  j++;
  setTimeout(reveal, step);
}}
typeChar();
"""
    return _shell(payload.get("title", "terminal"), body, TERMINAL_CSS, script)


# --------------------------------------------------------------------------
COMMANDS_BY_KEY = {
    "pytest": "uv run pytest",
    "ruff": "uv run ruff check .",
    "train": "uv run python scripts/train.py",
    "shap_layers": "uv run python scripts/explain_demo.py --borderline --no-llm",
    "agent_payload": "uv run python scripts/explain_demo.py --borderline --no-llm --json",
    "demo_raw": "uv run python scripts/explain_demo.py --borderline --raw",
    "demo_verified": "uv run python scripts/explain_demo.py --borderline",
}


def render(visual, duration: float) -> str:
    """Bir Visual için HTML üretir (kind: card | figure | terminal)."""
    if visual.kind == "card":
        return card_html(visual.payload, duration)
    if visual.kind == "figure":
        return figure_html(visual.payload, duration)
    if visual.kind == "terminal":
        cmd = COMMANDS_BY_KEY.get(visual.payload["output_key"], "uv run ...")
        return terminal_html(visual.payload, duration, cmd)
    raise ValueError(f"HTML sahnesi değil: {visual.kind}")
