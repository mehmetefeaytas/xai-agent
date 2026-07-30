"""Türkçe seslendirme: macOS ``say`` + Yelda sesi.

Yelda, sistemde kurulu tek Türkçe ses (``say -v '?' | grep tr_TR``). Sahne
başına bir ses dosyası üretilir, süresi ``ffprobe`` ile ölçülür ve zaman
çizelgesine yazılır. Görüntü sonradan bu sürelere hizalanır — böylece anlatım
ile ekrandaki hareket hiç kaymaz.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from narration import Scene

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
VOICE = "Yelda"
RATE = 172  # kelime/dakika — 165 fazla yavaş, 185 aceleci duruyor
SAMPLE_RATE = 48000


@dataclass
class AudioClip:
    scene_id: str
    path: Path
    duration: float


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def render_scene_audio(scene: Scene, out_dir: Path, force: bool = False) -> AudioClip:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{scene.id}.wav"
    txt = out_dir / f"{scene.id}.txt"
    txt.write_text(scene.spoken, encoding="utf-8")

    if force or not wav.exists():
        aiff = out_dir / f"{scene.id}.aiff"
        # -f ile dosyadan okuyoruz: kabuk tırnak/kaçış sorunu hiç yaşanmıyor.
        subprocess.run(
            ["say", "-v", VOICE, "-r", str(RATE), "-f", str(txt), "-o", str(aiff)],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(aiff),
                "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(wav),
            ],
            check=True,
        )
        aiff.unlink(missing_ok=True)

    return AudioClip(scene.id, wav, _ffprobe_duration(wav))


def render_all(scenes: list[Scene], video_key: str, force: bool = False) -> list[AudioClip]:
    out_dir = BUILD / "audio" / video_key
    clips = [render_scene_audio(s, out_dir, force=force) for s in scenes]
    total = sum(c.duration for c in clips)
    timeline = {
        "video": video_key,
        "voice": VOICE,
        "rate": RATE,
        "total_seconds": round(total, 2),
        "scenes": [
            {"id": c.scene_id, "seconds": round(c.duration, 2), "wav": c.path.name}
            for c in clips
        ],
    }
    path = BUILD / f"timeline-{video_key}.json"
    path.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"  {len(clips)} ses klibi · toplam {total / 60:.1f} dakika "
        f"({total:.0f} sn) · {path.name}"
    )
    return clips


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(scenes: list[Scene], durations: list[float], path: Path) -> None:
    """Altyapı sesle aynı kaynaktan üretildiği için altyazı asla kaymaz.

    Uzun anlatımlar iki alt bloğa bölünür; tek bir altyazı karesinde 25 sn
    metin okunamaz.
    """
    blocks: list[str] = []
    index = 1
    cursor = 0.0
    for scene, dur in zip(scenes, durations):
        chunks = _split_for_subtitle(scene.text)
        weights = [len(c) for c in chunks]
        total_w = sum(weights) or 1
        start = cursor
        for chunk, w in zip(chunks, weights):
            span = dur * w / total_w
            blocks.append(
                f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(start + span)}\n"
                f"{chunk}\n"
            )
            index += 1
            start += span
        cursor += dur
    path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"  altyazı: {path.name} ({index - 1} blok, {cursor:.0f} sn)")


def _split_for_subtitle(text: str, max_chars: int = 150) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > max_chars:
            chunks.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]
