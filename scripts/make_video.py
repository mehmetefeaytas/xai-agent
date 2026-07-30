"""Tanıtım videolarını uçtan uca üretir.

Hat: anlatım metni → Türkçe seslendirme (süre ölçümü) → sahne kaydı
(Playwright) → ses/görüntü hizalama → birleştirme → mp4 + srt + gif.

Kullanım
--------
    uv run python scripts/make_video.py --sesler          # yalnızca seslendirme
    uv run python scripts/make_video.py --ekran-goruntusu # README için UI çekimi
    uv run python scripts/make_video.py tanitim           # 90 saniyelik sürüm
    uv run python scripts/make_video.py derin             # derin anlatım
    uv run python scripts/make_video.py derin --yeniden-cek v_ui_explain

Ön koşullar: Streamlit ``localhost:8520``'de ve Ollama ``localhost:11434``'te
ayakta olmalı; ``build/outputs/`` altındaki gerçek komut çıktıları hazır olmalı
(``scripts/video/capture_outputs.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "video"))

import assemble  # noqa: E402
import capture  # noqa: E402
import tts  # noqa: E402
from narration import VIDEOS  # noqa: E402

BUILD = ROOT / "build"
VIDEO_DIR = ROOT / "docs" / "video"
IMAGE_DIR = ROOT / "docs" / "images"


def build_video(key: str, force_audio: bool = False,
                recapture: set[str] | None = None) -> Path:
    spec = VIDEOS[key]
    scenes = spec["scenes"]
    print(f"\n=== {spec['title']} ({len(scenes)} sahne) ===")

    print("1) Türkçe seslendirme")
    clips_audio = tts.render_all(scenes, key, force=force_audio)
    durations = [c.duration for c in clips_audio]

    print("2) Sahne kaydı (Playwright)")
    clips = capture.capture_scenes(scenes, durations, only=recapture)

    print("3) Ses/görüntü hizalama")
    parts: list[Path] = []
    report = []
    for scene, audio in zip(scenes, clips_audio):
        part = BUILD / "parts" / key / f"{scene.id}.mp4"
        info = assemble.build_scene(clips[scene.id], audio.path, part)
        parts.append(part)
        flag = f"  ×{info['speed']:.1f} hızlandırıldı" if info["speed"] >= 1.05 else ""
        print(f"   {scene.id:<22} görüntü {info['video']:5.1f} → ses "
              f"{info['audio']:5.1f} sn{flag}")
        report.append({"scene": scene.id, **info})

    print("4) Birleştirme")
    out = VIDEO_DIR / f"{spec['output']}.mp4"
    assemble.concat(parts, out)

    print("5) Altyazı")
    tts.write_srt(scenes, durations, VIDEO_DIR / f"{spec['output']}.srt")

    health = assemble.describe(out)
    print(f"\n   {out.relative_to(ROOT)}")
    print(f"   süre {health['duration'] / 60:.1f} dk · {health['mb']:.1f} MB · "
          f"{health['video']} · ses: {health['audio']}")

    (BUILD / f"assemble-{key}.json").write_text(
        json.dumps({"scenes": report, "health": health}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="XAI-Agent tanıtım videoları")
    parser.add_argument("videos", nargs="*", choices=list(VIDEOS),
                        help="üretilecek videolar (varsayılan: hepsi)")
    parser.add_argument("--sesler", action="store_true",
                        help="yalnızca seslendirmeyi üret ve süreleri yazdır")
    parser.add_argument("--yeniden-ses", action="store_true",
                        help="ses dosyalarını yeniden üret (metin değiştiyse)")
    parser.add_argument("--yeniden-cek", nargs="*", metavar="VISUAL_ID",
                        help="bu görselleri önbellekten silip yeniden kaydet")
    parser.add_argument("--ekran-goruntusu", action="store_true",
                        help="README için beş sekmenin ekran görüntüsünü al")
    parser.add_argument("--gif", metavar="SANIYE", type=float, default=None,
                        help="tanıtım videosundan bu saniyeden başlayan GIF üret")
    args = parser.parse_args()

    if args.ekran_goruntusu:
        print("README ekran görüntüleri (canlı LLM gerekir)")
        capture.capture_ui_screenshots(IMAGE_DIR / "ui")
        return 0

    if args.sesler:
        for key, spec in VIDEOS.items():
            print(f"\n=== {spec['title']} ===")
            tts.render_all(spec["scenes"], key, force=args.yeniden_ses)
        return 0

    selected = args.videos or list(VIDEOS)
    recapture = set(args.yeniden_cek) if args.yeniden_cek else None
    if recapture:
        for vid in recapture:
            (BUILD / "clips" / f"{vid}.webm").unlink(missing_ok=True)

    produced: dict[str, Path] = {}
    for key in selected:
        produced[key] = build_video(key, force_audio=args.yeniden_ses,
                                    recapture=recapture)

    if args.gif is not None and "tanitim" in produced:
        gif = IMAGE_DIR / "tanitim.gif"
        assemble.make_gif(produced["tanitim"], gif, start=args.gif, duration=9.0)
        print(f"\n   {gif.relative_to(ROOT)} · {gif.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
