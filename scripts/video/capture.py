"""Sahneleri Playwright ile video klibine çevirir.

Tek yakalama mekanizması: ``record_video_dir``. Hem gerçek Streamlit
uygulaması hem üretilen HTML sayfaları aynı hattan geçtiği için bütün klipler
aynı çözünürlük ve kare hızında oluyor.

Klipler ``visual.id`` ile önbelleklenir: iki video aynı görseli paylaşabilir
(örneğin 90 saniyelik tanıtım ile derin anlatım aynı şelale slaytını kullanır)
ve pahalı olan canlı LLM sahnesi bir kez çekilir.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import html_scenes
import ui_scenes
from narration import Scene
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "build" / "clips"
WIDTH, HEIGHT = 1280, 720


def _record_one(browser, scene: Scene, duration: float, target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="xai-vid-") as tmp:
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=tmp,
            record_video_size={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
        )
        page = context.new_page()
        try:
            if scene.visual.kind == "ui":
                ui_scenes.run(page, scene.visual, duration)
            else:
                page.set_content(html_scenes.render(scene.visual, duration))
                page.wait_for_timeout(int(duration * 1000) + 300)
        finally:
            page.close()
            context.close()

        produced = sorted(Path(tmp).glob("*.webm"))
        if not produced:
            raise RuntimeError(f"{scene.visual.id}: Playwright video üretmedi")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced[0]), target)


def capture_scenes(
    scenes: list[Scene],
    durations: list[float],
    force: bool = False,
    only: set[str] | None = None,
) -> dict[str, Path]:
    """Her sahne için bir klip üretir; var olanı yeniden çekmez."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    todo: list[tuple[Scene, float]] = []
    for scene, dur in zip(scenes, durations):
        clip = CLIPS / f"{scene.visual.id}.webm"
        result[scene.id] = clip
        if only and scene.visual.id not in only:
            continue
        if clip.exists() and not force:
            continue
        todo.append((scene, dur))

    # Aynı görseli iki video paylaşıyorsa iki kez çekmeyelim.
    seen: set[str] = set()
    unique = []
    for scene, dur in todo:
        if scene.visual.id in seen:
            continue
        seen.add(scene.visual.id)
        unique.append((scene, dur))

    if not unique:
        print("  (tüm klipler önbellekte)")
        return result

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for i, (scene, dur) in enumerate(unique, 1):
                clip = CLIPS / f"{scene.visual.id}.webm"
                kind = scene.visual.kind
                print(
                    f"  [{i}/{len(unique)}] {scene.visual.id:<22} {kind:<9} "
                    f"hedef {dur:5.1f} sn ...",
                    end="",
                    flush=True,
                )
                _record_one(browser, scene, dur, clip)
                kb = clip.stat().st_size / 1024
                print(f" ✓ {kb:.0f} KB")
        finally:
            browser.close()
    return result


def capture_ui_screenshots(out_dir: Path) -> list[str]:
    """README için beş sekmenin ekran görüntüsünü alır (video değil)."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context(
                viewport={"width": 1500, "height": 1000}, device_scale_factor=2
            )
            page = context.new_page()
            saved = ui_scenes.take_screenshots(page, out_dir)
            context.close()
        finally:
            browser.close()
    return saved
