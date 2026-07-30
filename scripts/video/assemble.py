"""Klipleri sesle hizalar, birleştirir ve mp4 / GIF üretir.

Hizalama kuralı basit ve her sahnede aynı:

* Görüntü sesten KISAysa son kare tutulur (``tpad``) — donmuş bir kare,
  anlatım biterken kesilen bir görüntüden iyidir.
* Görüntü sesten UZUNsa hızlandırılır (``setpts``). Canlı LLM sahnelerinde
  bu 2-3 katına çıkabiliyor; izleyiciyi yanıltmamak için ekrana
  "×N hızlandırıldı" rozeti basılır.

Rozet neden resim? Bu ffmpeg derlemesinde ``libass``/``freetype`` yok, yani
``drawtext``/``subtitles`` filtreleri kullanılamıyor. Rozeti PIL ile PNG olarak
üretip ``overlay`` filtresiyle bindirmek aynı işi görüyor. Altyazı da bu yüzden
videoya gömülmüyor, yanında ``.srt`` olarak veriliyor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BADGES = BUILD / "badges"

WIDTH, HEIGHT, FPS = 1280, 720, 30
#: Bu orandan fazla hızlandırma varsa izleyiciye söylemek zorunludur.
BADGE_THRESHOLD = 1.25


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError("ffmpeg hatası:\n" + "\n".join(tail))


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _badge(factor: float) -> Path:
    """"×2.4 hızlandırıldı" rozetini PNG olarak üretir (ffmpeg'de drawtext yok)."""
    from PIL import Image, ImageDraw, ImageFont

    BADGES.mkdir(parents=True, exist_ok=True)
    label = f"×{factor:.1f} hızlandırıldı"
    path = BADGES / f"badge-{factor:.1f}.png"
    if path.exists():
        return path

    font = None
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 22)
            break
    if font is None:  # son çare
        font = ImageFont.load_default()

    pad_x, pad_y = 18, 11
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), label, font=font)
    w, h = box[2] - box[0] + pad_x * 2, box[3] - box[1] + pad_y * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=9, fill=(11, 18, 32, 220),
                           outline=(77, 163, 255, 235), width=2)
    draw.text((pad_x - box[0], pad_y - box[1]), label, font=font,
              fill=(232, 238, 247, 255))
    img.save(path)
    return path


def build_scene(clip: Path, wav: Path, out: Path) -> dict[str, float]:
    """Bir klibi sesiyle hizalayıp mp4'e kodlar."""
    v_dur = probe_duration(clip)
    a_dur = probe_duration(wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    inputs = ["-i", str(clip), "-i", str(wav)]
    factor = 1.0
    chain = [f"fps={FPS}", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease",
             f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=#0b1220", "setsar=1"]

    if v_dur > a_dur + 0.15:
        factor = v_dur / a_dur
        chain.append(f"setpts=PTS/{factor:.6f}")
    elif a_dur > v_dur + 0.05:
        chain.append(f"tpad=stop_mode=clone:stop_duration={a_dur - v_dur + 0.30:.3f}")

    badge_idx = None
    if factor >= BADGE_THRESHOLD:
        inputs += ["-i", str(_badge(factor))]
        badge_idx = 2

    filter_parts = [f"[0:v]{','.join(chain)}[v0]"]
    if badge_idx is not None:
        filter_parts.append(f"[v0][{badge_idx}:v]overlay=W-w-28:28[v]")
        vmap = "[v]"
    else:
        filter_parts.append("[v0]null[v]")
        vmap = "[v]"

    _run([
        "ffmpeg", "-v", "error", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", vmap, "-map", "1:a",
        "-t", f"{a_dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "26",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(out),
    ])
    return {"video": v_dur, "audio": a_dur, "speed": factor}


def concat(parts: list[Path], out: Path) -> None:
    """Aynı parametrelerle kodlanmış parçaları yeniden kodlamadan birleştirir."""
    listing = BUILD / f"concat-{out.stem}.txt"
    listing.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out),
    ])


def make_gif(src: Path, out: Path, start: float, duration: float,
             width: int = 760, fps: int = 12) -> None:
    """README'nin en üstünde inline oynayacak kısa döngü."""
    palette = BUILD / "gif-palette.png"
    common = f"fps={fps},scale={width}:-1:flags=lanczos"
    _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{duration}",
          "-i", str(src), "-vf", f"{common},palettegen=stats_mode=diff",
          str(palette)])
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{duration}",
          "-i", str(src), "-i", str(palette),
          "-lavfi", f"{common}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
          "-loop", "0", str(out)])


def describe(path: Path) -> dict[str, object]:
    """Kodlanmış videonun sağlık raporu — ses akışı var mı, süre ne?"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    import json

    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), {})
    audio = next((s for s in streams if s["codec_type"] == "audio"), {})
    return {
        "duration": float(data["format"]["duration"]),
        "mb": int(data["format"]["size"]) / 1e6,
        "video": f"{video.get('codec_name')} {video.get('width')}x{video.get('height')}"
                 f" @ {video.get('r_frame_rate')}",
        "audio": audio.get("codec_name") or "SES YOK",
        "has_audio": bool(audio),
    }
