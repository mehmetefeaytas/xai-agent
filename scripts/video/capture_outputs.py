"""Videoda gösterilecek terminal çıktılarını GERÇEKTEN çalıştırıp kaydeder.

Videodaki terminal sahneleri uydurma değil: burada yakalanan gerçek stdout,
sonradan daktilo efektiyle yeniden oynatılıyor. Çıktılar ``build/outputs/``
altına yazılır.

LLM içeren komutlar yavaştır (yerel 7B model, soru başına 40-70 sn) ve Ollama
eşzamanlı yükte çökebiliyor — bu yüzden komutlar SIRAYLA çalıştırılır.

Kullanım
--------
    uv run python scripts/video/capture_outputs.py            # hepsi
    uv run python scripts/video/capture_outputs.py --fast     # LLM'siz olanlar
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "build" / "outputs"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

# (anahtar, gösterilecek komut, gerçek argv, LLM gerekiyor mu)
COMMANDS: list[tuple[str, str, list[str], bool]] = [
    (
        "pytest",
        "uv run pytest",
        ["uv", "run", "pytest"],
        False,
    ),
    (
        "ruff",
        "uv run ruff check .",
        ["uv", "run", "ruff", "check", "."],
        False,
    ),
    (
        "train",
        "uv run python scripts/train.py",
        ["uv", "run", "python", "scripts/train.py"],
        False,
    ),
    (
        "shap_layers",
        "uv run python scripts/explain_demo.py --borderline --no-llm",
        ["uv", "run", "python", "scripts/explain_demo.py", "--borderline", "--no-llm"],
        False,
    ),
    (
        "agent_payload",
        "uv run python scripts/explain_demo.py --borderline --no-llm --json",
        ["uv", "run", "python", "scripts/explain_demo.py", "--borderline",
         "--no-llm", "--json"],
        False,
    ),
    (
        "demo_raw",
        "uv run python scripts/explain_demo.py --borderline --raw",
        ["uv", "run", "python", "scripts/explain_demo.py", "--borderline", "--raw"],
        True,
    ),
    (
        "demo_verified",
        "uv run python scripts/explain_demo.py --borderline",
        ["uv", "run", "python", "scripts/explain_demo.py", "--borderline"],
        True,
    ),
]


def run_one(key: str, shown: str, argv: list[str]) -> dict[str, object]:
    print(f"\n>>> [{key}] {shown}", flush=True)
    started = time.monotonic()
    proc = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, timeout=2400
    )
    elapsed = time.monotonic() - started
    body = ANSI_RE.sub("", (proc.stdout or "") + (proc.stderr or ""))
    body = body.replace("\r", "")
    (OUT_DIR / f"{key}.txt").write_text(body, encoding="utf-8")
    print(
        f"    çıkış kodu {proc.returncode}, {len(body.splitlines())} satır, "
        f"{elapsed:.0f} sn",
        flush=True,
    )
    return {
        "key": key,
        "command": shown,
        "returncode": proc.returncode,
        "lines": len(body.splitlines()),
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal çıktılarını yakala")
    parser.add_argument("--fast", action="store_true", help="LLM gerektirenleri atla")
    parser.add_argument("--only", nargs="*", help="Yalnızca bu anahtarlar")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = [
        c for c in COMMANDS
        if (not args.fast or not c[3]) and (not args.only or c[0] in args.only)
    ]

    manifest = []
    for key, shown, argv, _needs_llm in selected:
        try:
            manifest.append(run_one(key, shown, argv))
        except subprocess.TimeoutExpired:
            print(f"    ZAMAN AŞIMI — {key} atlandı", flush=True)
            manifest.append({"key": key, "command": shown, "returncode": "timeout"})

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{len(manifest)} çıktı {OUT_DIR} altına yazıldı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
