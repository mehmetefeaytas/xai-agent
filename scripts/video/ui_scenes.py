"""Canlı Streamlit arayüzünü Playwright ile sürer.

Buradaki hiçbir şey taklit değil: gerçek uygulama, gerçek tıklamalar, gerçek
LightGBM tahmini ve gerçek yerel LLM. Üretim 40-70 saniye sürebiliyor; bu
bekleme videoda anlatımla doldurulur, artan ölü zaman kurguda hızlandırılır.

Seçiciler tahmin edilmedi, uygulamanın DOM'u üzerinden doğrulandı:
sekmeler ``role=tab``, birincil düğmeler metinle, kenar çubuğu açılır
menüleri ``[data-testid="stSelectbox"]`` + etiket metni.
"""

from __future__ import annotations

from playwright.sync_api import Page

APP_URL = "http://localhost:8520"

#: Streamlit'in kendi kabuğu (üst bar, Deploy düğmesi, öğe araç çubukları)
#: videoda dikkat dağıtıyor ve projeyle ilgisi yok.
HIDE_CHROME = """
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stAppDeployButton, #MainMenu, footer,
[data-testid="stElementToolbar"] { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1.2rem !important; }
"""

GENERATION_TIMEOUT = 240_000


def open_app(page: Page) -> None:
    page.goto(APP_URL, wait_until="networkidle")
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
    page.wait_for_timeout(2500)
    page.add_style_tag(content=HIDE_CHROME)
    page.wait_for_timeout(400)


def _sidebar_select(page: Page, label: str, option: str) -> None:
    """Etiketiyle bulunan açılır menüden bir seçenek seçer."""
    box = page.locator(
        f'[data-testid="stSidebar"] [data-testid="stSelectbox"]:has-text("{label}")'
    ).first
    box.locator('[role="combobox"]').click()
    page.wait_for_timeout(650)
    page.get_by_role("option", name=option, exact=False).first.click()
    page.wait_for_timeout(2600)


def pick_borderline_applicant(page: Page) -> None:
    """Eşiğe en yakın başvuruyu seçer — A-006, figürlerdekiyle aynı satır."""
    _sidebar_select(page, "Filtre", "Eşiğe en yakınlar (kıl payı)")
    _sidebar_select(page, "Başvuru", "A-006")


def _tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1400)


def _hold(page: Page, target_ms: float, elapsed_ms: float) -> None:
    """Anlatım bitene kadar sahneyi ayakta tut."""
    remaining = target_ms - elapsed_ms
    if remaining > 0:
        page.wait_for_timeout(int(remaining))


# --------------------------------------------------------------------------
# Sahne eylemleri
# --------------------------------------------------------------------------
def action_explain(page: Page, duration: float) -> None:
    """Denetimli modda canlı açıklama üretir ve sonucu gezdirir."""
    open_app(page)
    pick_borderline_applicant(page)

    toggle = page.get_by_role("switch", name="🛡️ Denetimli mod (critic-and-revise)")
    if not toggle.is_checked():
        toggle.click()
        page.wait_for_timeout(1800)

    page.get_by_role("button", name="🗣️ Açıklamayı üret").click()
    # Denetim sonucu satırı ("✅ Sadakat denetimi geçti" veya "⚠️ ... ihlal buldu")
    # üretimin bittiğinin tek güvenilir işareti.
    page.get_by_text("Sadakat denetimi").first.wait_for(timeout=GENERATION_TIMEOUT)
    page.wait_for_timeout(1500)

    page.get_by_text("🔧 Ajan hangi tool'ları çağırdı?").first.click()
    page.wait_for_timeout(2200)

    # Anlatıyı ve yanındaki şelaleyi tarayarak göster
    for offset in (280, 560, 240, 0):
        page.mouse.wheel(0, offset if offset else -1200)
        page.wait_for_timeout(1500)

    _hold(page, duration * 1000, 0)


def action_whatif(page: Page, duration: float) -> None:
    """Karşı-olgusal senaryo: vadeyi kısaltıp modeli yeniden koşturur."""
    open_app(page)
    pick_borderline_applicant(page)
    _tab(page, "🔀 What-if")

    feature_box = page.locator(
        '[data-testid="stSelectbox"]:has-text("Değiştirilecek özellik")'
    ).first
    feature_box.locator('[role="combobox"]').click()
    page.wait_for_timeout(700)
    page.get_by_role("option", name="Kredi vadesi", exact=True).first.click()
    page.wait_for_timeout(3000)

    # Sayısal özellik → kaydırıcı. DİKKAT: bu Streamlit sürümünde kaydırıcının
    # tutamağı ``role="slider"`` TAŞIMIYOR; odaklanabilir bir <input>.
    # Seçici tahmin edilmedi, DOM'dan doğrulandı.
    slider = page.locator('[data-testid="stSlider"] input').first
    if slider.count() > 0:
        slider.focus()
        # Ok tuşlarıyla adım adım azaltmak videoda da anlaşılır: 18 → 12 ay.
        for _ in range(6):
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(400)
    page.wait_for_timeout(1600)

    page.get_by_role("button", name="▶️ Senaryoyu çalıştır").click()
    page.wait_for_timeout(3500)
    page.mouse.wheel(0, 320)
    page.wait_for_timeout(2000)

    _hold(page, duration * 1000, 0)


def action_tour(page: Page, duration: float) -> None:
    """Beş sekmeyi sırayla gezer."""
    open_app(page)
    pick_borderline_applicant(page)
    per_tab = max(1800, (duration * 1000 - 6000) / 5)
    for name in ("🗣️ Açıklama", "📊 Etkenler", "🔀 What-if", "💬 Sohbet", "🧪 Denetim"):
        _tab(page, name)
        page.wait_for_timeout(int(per_tab))


ACTIONS = {
    "explain": action_explain,
    "whatif": action_whatif,
    "tour": action_tour,
}


def run(page: Page, visual, duration: float) -> None:
    action = visual.payload["action"]
    if action not in ACTIONS:
        raise ValueError(f"Bilinmeyen UI eylemi: {action}")
    ACTIONS[action](page, duration)


# --------------------------------------------------------------------------
# README için ekran görüntüleri (video değil)
# --------------------------------------------------------------------------
TABS_FOR_SHOTS = (
    ("01-aciklama", "🗣️ Açıklama"),
    ("02-etkenler", "📊 Etkenler"),
    ("03-whatif", "🔀 What-if"),
    ("04-sohbet", "💬 Sohbet"),
    ("05-denetim", "🧪 Denetim"),
)


def take_screenshots(page: Page, out_dir) -> list[str]:
    """Beş sekmenin ekran görüntüsünü alır; Açıklama sekmesi canlı üretimle."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    open_app(page)
    pick_borderline_applicant(page)

    # 1) Açıklama — gerçek bir ajan yanıtıyla anlamlı olsun
    page.get_by_role("button", name="🗣️ Açıklamayı üret").click()
    page.get_by_text("Sadakat denetimi").first.wait_for(timeout=GENERATION_TIMEOUT)
    page.wait_for_timeout(2000)
    page.get_by_text("🔧 Ajan hangi tool'ları çağırdı?").first.click()
    page.wait_for_timeout(1500)

    for name, tab in TABS_FOR_SHOTS:
        _tab(page, tab)
        if tab == "🔀 What-if":
            # Boş bir senaryo ekranı bilgi vermez: bir sonuç üretip çekelim.
            try:
                page.get_by_role("button", name="▶️ Senaryoyu çalıştır").click()
                page.wait_for_timeout(3000)
            except Exception:  # noqa: BLE001
                pass
        page.wait_for_timeout(900)
        path = out_dir / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        saved.append(path.name)
        print(f"  ✓ {path.name}")
    return saved
