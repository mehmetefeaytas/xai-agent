"""Videonun tek doğruluk kaynağı: sahneler, anlatım metni ve görsel tarifi.

Ses (``speech``) ile altyazı (``text``) ayrı tutulur. Sebebi somut: sistemdeki
tek Türkçe ses olan Yelda, "SHAP", "LightGBM", "AUC" gibi terimleri harf harf
veya İngilizce okuyamıyor. Altyazıda doğru yazım, seslendirmede fonetik yazım
kullanılır — ikisi de aynı yerden üretildiği için kaymaları imkânsız.

Sahnelerin ``visual`` alanı, hangi görüntünün kaydedileceğini söyler. Aynı
``visual_id`` iki videoda da geçebilir; klip bir kez kaydedilip yeniden
kullanılır (bkz. capture.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Visual:
    """Bir sahnede ekranda ne görünecek."""

    id: str
    kind: str  # "card" | "terminal" | "figure" | "ui"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    id: str
    text: str  # altyazıda görünen, doğru yazımlı metin
    visual: Visual
    speech: str | None = None  # TTS'e giden fonetik metin (None → text)

    @property
    def spoken(self) -> str:
        return self.speech or self.text


# --------------------------------------------------------------------------
# Görseller
# --------------------------------------------------------------------------
def card(vid: str, kicker: str, title: str, lines: list[str]) -> Visual:
    return Visual(vid, "card", {"kicker": kicker, "title": title, "lines": lines})


def figure(vid: str, image: str, caption: str, zoom: dict | None = None) -> Visual:
    return Visual(vid, "figure", {"image": image, "caption": caption, "zoom": zoom})


def terminal(
    vid: str,
    output_key: str,
    title: str = "xai-agent — zsh",
    excerpt: tuple[str, str] | None = None,
    max_lines: int | None = None,
) -> Visual:
    return Visual(
        vid,
        "terminal",
        {
            "output_key": output_key,
            "title": title,
            "excerpt": excerpt,
            "max_lines": max_lines,
        },
    )


def ui(vid: str, action: str, **kw: Any) -> Visual:
    return Visual(vid, "ui", {"action": action, **kw})


# --------------------------------------------------------------------------
# DERİN ANLATIM (~9 dakika)
# --------------------------------------------------------------------------
DEEP: list[Scene] = [
    Scene(
        id="d01_baslik",
        visual=card(
            "v_baslik",
            "Mezuniyet / portfolyo projesi · Mehmet Efe Aytaş",
            "XAI-Agent",
            [
                "Şeffaf kredi risk ajanı",
                "LightGBM → SHAP → LLM ajanı → sadakat denetçisi",
            ],
        ),
        text=(
            "Merhaba. Bu, XAI-Agent adlı projenin ayrıntılı anlatımı. "
            "Amacı tek cümleyle şu: bir kredi başvurusunun neden reddedildiğini, "
            "makine öğrenmesi modelinin gerçek matematiğine sadık kalarak, "
            "sıradan bir insanın anlayacağı Türkçeyle anlatmak. "
            "Üç katmandan oluşuyor ve üstlerinde dördüncü bir katman var: "
            "anlatılanların doğruluğunu programatik olarak denetleyen bir denetçi."
        ),
        speech=(
            "Merhaba. Bu, eks ey ay ejent adlı projenin ayrıntılı anlatımı. "
            "Amacı tek cümleyle şu: bir kredi başvurusunun neden reddedildiğini, "
            "makine öğrenmesi modelinin gerçek matematiğine sadık kalarak, "
            "sıradan bir insanın anlayacağı Türkçeyle anlatmak. "
            "Üç katmandan oluşuyor ve üstlerinde dördüncü bir katman var: "
            "anlatılanların doğruluğunu programatik olarak denetleyen bir denetçi."
        ),
    ),
    Scene(
        id="d02_problem",
        visual=card(
            "v_problem",
            "Problem",
            "«Reddedildi» yetmez",
            [
                "Kredi kararı hukuken gerekçe ister",
                "Ama LLM'e «bu kararı açıkla» demek uydurma üretir",
                "Gerekçe modelin matematiğinden gelmeli, dilden değil",
            ],
        ),
        text=(
            "Problem şu: bir kredi başvurusu reddedildiğinde, başvuru sahibinin "
            "gerekçe hakkı var. Ama modele bakıp «bunu açıkla» demek yetmez, "
            "çünkü dil modelleri ikna edici ama yanlış gerekçeler üretmekte "
            "çok iyidir. Bu projenin kurucu ilkesi şudur: kararı model verir, "
            "gerekçeyi SHAP hesaplar, dili ajan kurar — ve ajanın kurduğu her "
            "cümle, SHAP'ın sayılarına karşı otomatik olarak denetlenir."
        ),
        speech=(
            "Problem şu: bir kredi başvurusu reddedildiğinde, başvuru sahibinin "
            "gerekçe hakkı var. Ama modele bakıp «bunu açıkla» demek yetmez, "
            "çünkü dil modelleri ikna edici ama yanlış gerekçeler üretmekte "
            "çok iyidir. Bu projenin kurucu ilkesi şudur: kararı model verir, "
            "gerekçeyi şap hesaplar, dili ajan kurar — ve ajanın kurduğu her "
            "cümle, şapın sayılarına karşı otomatik olarak denetlenir."
        ),
    ),
    Scene(
        id="d03_veri",
        visual=card(
            "v_veri",
            "Veri",
            "UCI German Credit",
            [
                "1000 başvuru · 20 özellik · 800 eğitim / 200 test",
                "Hedef: 1 = riskli (bad) · sınıf dengesi %30",
                "Maliyet matrisi veri setiyle birlikte gelir: yanlış kabul 5, yanlış ret 1",
                "personal_status ve foreign_worker modelden ÇIKARILDI · yaş tutuldu",
            ],
        ),
        text=(
            "Veri seti UCI German Credit: bin başvuru, yirmi özellik, sekiz yüz "
            "eğitim ve iki yüz test satırı. Bu veri setinin özel bir yanı var — "
            "kendi maliyet matrisiyle geliyor. Riskli bir müşteriyi onaylamak, "
            "iyi bir müşteriyi reddetmekten beş kat pahalı sayılıyor. "
            "Cinsiyet bilgisi taşıyan medeni durum ve yabancı işçi alanlarını "
            "modelden çıkardım; yaşı bilinçli olarak bıraktım, çünkü sonra "
            "adalet denetiminde bunun bedelini ölçeceğiz."
        ),
        speech=(
            "Veri seti u ce i cörman kredit: bin başvuru, yirmi özellik, sekiz yüz "
            "eğitim ve iki yüz test satırı. Bu veri setinin özel bir yanı var — "
            "kendi maliyet matrisiyle geliyor. Riskli bir müşteriyi onaylamak, "
            "iyi bir müşteriyi reddetmekten beş kat pahalı sayılıyor. "
            "Cinsiyet bilgisi taşıyan medeni durum ve yabancı işçi alanlarını "
            "modelden çıkardım; yaşı bilinçli olarak bıraktım, çünkü sonra "
            "adalet denetiminde bunun bedelini ölçeceğiz."
        ),
    ),
    Scene(
        id="d04_egitim",
        visual=terminal("v_train", "train", max_lines=46),
        text=(
            "Birinci katman: LightGBM. Eğitimi canlı çalıştırıyorum. "
            "Kategorik değişkenleri one-hot'a açmıyorum — LightGBM'in kendi "
            "kategorik desteğini kullanıyorum. Bunun tek sebebi hız değil: "
            "one-hot açarsanız bir özelliğin SHAP değeri onlarca parçaya bölünür "
            "ve «vadesiz hesap durumu kararı şu kadar etkiledi» diyemezsiniz. "
            "Model iki yüz kırk üç ağaçta duruyor ve bu sayı tek bir doğrulama "
            "kümesinde değil, beş katın ortalamasında belirlendi."
        ),
        speech=(
            "Birinci katman: layt ci bi em. Eğitimi canlı çalıştırıyorum. "
            "Kategorik değişkenleri van hot'a açmıyorum — layt ci bi em'in kendi "
            "kategorik desteğini kullanıyorum. Bunun tek sebebi hız değil: "
            "van hot açarsanız bir özelliğin şap değeri onlarca parçaya bölünür "
            "ve «vadesiz hesap durumu kararı şu kadar etkiledi» diyemezsiniz. "
            "Model iki yüz kırk üç ağaçta duruyor ve bu sayı tek bir doğrulama "
            "kümesinde değil, beş katın ortalamasında belirlendi."
        ),
    ),
    Scene(
        id="d05_cv",
        visual=figure(
            "v_cv", "10-cv-katlamalari.png",
            "5 katlı çapraz doğrulama — katlar arası fark 0.106",
        ),
        text=(
            "Neden beş kat? Çünkü bin satırlık bir veri setinde tek bir skora "
            "güvenilemez. Bakın: katlar arasındaki fark yüzde on. Dördüncü kat "
            "yetmiş bir, ikinci kat seksen iki veriyor. İlk sürümde tek bir "
            "doğrulama kümesi kullanmıştım ve model otuz ağaçta durup yetersiz "
            "öğrenmişti. Çapraz doğrulamaya geçince hem model düzeldi hem de "
            "test skorunun şans eseri olmadığını gösterebildim."
        ),
    ),
    Scene(
        id="d06_baseline",
        visual=figure(
            "v_baseline", "02-roc-pr.png",
            "Dürüst karşılaştırma: lojistik regresyon AUC'de önde",
        ),
        text=(
            "Ve burada projenin en dürüst grafiği var. Lojistik regresyon, "
            "sıralama gücünde LightGBM'i geçiyor: sıfır virgül sekiz bir'e karşı "
            "sıfır virgül yedi dokuz. Bunu saklamıyorum, çünkü saklamak "
            "mühendislik değil pazarlama olur. LightGBM'i seçmemin sebebi başka: "
            "maliyette yüz dörde karşı yüz on beş, duyarlılıkta doksan ikiye "
            "karşı seksen sekiz. Yani riskli başvuruların daha azını kaçırıyor."
        ),
        speech=(
            "Ve burada projenin en dürüst grafiği var. Lojistik regresyon, "
            "sıralama gücünde layt ci bi em'i geçiyor: sıfır virgül sekiz bir'e karşı "
            "sıfır virgül yedi dokuz. Bunu saklamıyorum, çünkü saklamak "
            "mühendislik değil pazarlama olur. Layt ci bi em'i seçmemin sebebi başka: "
            "maliyette yüz dörde karşı yüz on beş, duyarlılıkta doksan ikiye "
            "karşı seksen sekiz. Yani riskli başvuruların daha azını kaçırıyor."
        ),
    ),
    Scene(
        id="d07_esik",
        visual=figure(
            "v_esik", "01-esik-maliyet.png",
            "Karar eşiği 0.50 değil 0.26 — maliyet asimetrisi",
        ),
        text=(
            "Şimdi çoğu projenin atladığı yer: karar eşiği. Varsayılan sıfır "
            "virgül beş, yanlış kabulün yanlış retten beş kat pahalı olduğu bir "
            "dünyada yanlış bir seçim. Maliyeti en küçük yapan eşik sıfır virgül "
            "yirmi altı. Teorik Bayes eşiği altıda bir, yani sıfır virgül on yedi; "
            "ampirik sonuç ona yakın. Kritik ayrıntı: eşiği yalnızca eğitim "
            "setinin kat-dışı tahminlerinde seçtim. Test setine eşik ayarlamak "
            "için asla dokunulmadı — gri eğri sadece doğrulama."
        ),
    ),
    Scene(
        id="d08_matris",
        visual=figure(
            "v_matris", "03-karmasiklik-matrisi.png",
            "İki eşik, iki sonuç: 16 kaçan riskli mi, 5 mi?",
        ),
        text=(
            "Bunun somut karşılığı bu iki tablo. Varsayılan eşikle isabet daha "
            "yüksek görünüyor: yüzde yetmiş iki. Ama on altı riskli başvuru "
            "onaylanıyor. Seçtiğim eşikle isabet yüzde elli sekize düşüyor, "
            "buna karşılık kaçan riskli sayısı beşe iniyor ve toplam maliyet "
            "yüz yirmiden yüz dörde geriliyor. İşte bu yüzden bu projede isabet "
            "oranı bir başarı ölçüsü olarak hiç kullanılmıyor — asimetrik "
            "maliyeti gizliyor."
        ),
    ),
    Scene(
        id="d09_shap",
        visual=terminal(
            "v_shap_terminal", "shap_layers",
            excerpt=("KATMAN 1", "Dışlanan korunan"), max_lines=40,
        ),
        text=(
            "İkinci katman: SHAP. Terminalde ilk iki katmanı birlikte görüyoruz. "
            "Karar, risk oranı, eşiğe uzaklık — sonra gerekçe. Riski artıran ve "
            "azaltan etkenler, her birinin toplam etkideki payıyla. "
            "SHAP değerlerini LightGBM'in kendi ağaç tabanlı uygulamasından "
            "alıyorum, ayrıca bağımsız SHAP kütüphanesiyle karşılaştırdım: "
            "iki hesap arasındaki fark tam olarak sıfır."
        ),
        speech=(
            "İkinci katman: şap. Terminalde ilk iki katmanı birlikte görüyoruz. "
            "Karar, risk oranı, eşiğe uzaklık — sonra gerekçe. Riski artıran ve "
            "azaltan etkenler, her birinin toplam etkideki payıyla. "
            "Şap değerlerini layt ci bi em'in kendi ağaç tabanlı uygulamasından "
            "alıyorum, ayrıca bağımsız şap kütüphanesiyle karşılaştırdım: "
            "iki hesap arasındaki fark tam olarak sıfır."
        ),
    ),
    Scene(
        id="d10_selale",
        visual=figure(
            "v_selale", "05-shap-selale.png",
            "A-006 · risk %25.9 · eşik %26 — kıl payı onay",
        ),
        text=(
            "Aynı açıklamanın görsel hali. Bu başvuru kıl payı onaylandı: risk "
            "yüzde yirmi beş virgül dokuz, eşik yüzde yirmi altı. Yeşiller riski "
            "azaltıyor, kırmızılar artırıyor. Dikkat edin: kredi geçmişinde "
            "«başka bankada devam eden kredi var» kaydı riski azaltıyor. "
            "Sezgiye aykırı ama veriye uygun — ödeme geçmişi kanıtlanmış müşteri "
            "daha güvenli. Model bunu veriden öğrendi, ben kural olarak yazmadım."
        ),
    ),
    Scene(
        id="d11_toplanabilirlik",
        visual=figure(
            "v_toplanabilirlik", "06-toplanabilirlik.png",
            "Toplanabilirlik: taban + Σ katkı = modelin çıktısı",
        ),
        text=(
            "Bu grafik projenin matematiksel omurgası. SHAP'ın yerel isabet "
            "aksiyomu şunu şart koşar: taban değer artı bütün katkıların toplamı, "
            "modelin çıktısına eşit olmak zorunda. İki yüz test başvurusunun "
            "hepsinde ölçtüm; en büyük sapma on üzeri eksi on beş mertebesinde. "
            "Bu bir yaklaşım hatası değil, kayan nokta aritmetiğinin sınırı. "
            "Yani açıklama, modelin yanında duran ikinci bir tahmin değil — "
            "modelin kendi kararının ayrıştırılmış hali."
        ),
        speech=(
            "Bu grafik projenin matematiksel omurgası. Şapın yerel isabet "
            "aksiyomu şunu şart koşar: taban değer artı bütün katkıların toplamı, "
            "modelin çıktısına eşit olmak zorunda. İki yüz test başvurusunun "
            "hepsinde ölçtüm; en büyük sapma on üzeri eksi on beş mertebesinde. "
            "Bu bir yaklaşım hatası değil, kayan nokta aritmetiğinin sınırı. "
            "Yani açıklama, modelin yanında duran ikinci bir tahmin değil — "
            "modelin kendi kararının ayrıştırılmış hali."
        ),
    ),
    Scene(
        id="d12_kopru",
        visual=terminal(
            "v_payload", "agent_payload",
            excerpt=("AJANA GİDEN YÜK", "}"), max_lines=44,
        ),
        text=(
            "Şimdi kritik tasarım kararı: SHAP ile ajan arasındaki köprü. "
            "Ajana giden yükte tek bir ham log-odds sayısı yok. Sadece "
            "kopyalanmaya hazır dizeler var: etki payı yüzde on dört nokta sıfır, "
            "etki yönü riski artırıyor, başvuru sahibi değiştirebilir mi hayır. "
            "Sebebi acı bir tecrübe: ilk sürümde ham katkı değerini gören model "
            "sıfır virgül seksen üçü «yüzde seksen üç risk» diye okudu. "
            "Modelin yorumlayabileceği hiçbir sayı bırakmayınca bu hata sınıfı "
            "tamamen ortadan kalktı."
        ),
        speech=(
            "Şimdi kritik tasarım kararı: şap ile ajan arasındaki köprü. "
            "Ajana giden yükte tek bir ham log ods sayısı yok. Sadece "
            "kopyalanmaya hazır dizeler var: etki payı yüzde on dört nokta sıfır, "
            "etki yönü riski artırıyor, başvuru sahibi değiştirebilir mi hayır. "
            "Sebebi acı bir tecrübe: ilk sürümde ham katkı değerini gören model "
            "sıfır virgül seksen üçü «yüzde seksen üç risk» diye okudu. "
            "Modelin yorumlayabileceği hiçbir sayı bırakmayınca bu hata sınıfı "
            "tamamen ortadan kalktı."
        ),
    ),
    Scene(
        id="d13_araclar",
        visual=card(
            "v_araclar",
            "Katman 3 — Ajan",
            "Dört tool, tek köprü",
            [
                "get_decision_explanation — kararı ve SHAP gerekçesini getirir",
                "run_what_if — modeli GERÇEKTEN yeniden çalıştırır",
                "get_feature_info — özelliğin ne olduğunu açıklar",
                "get_global_importance — modelin genel davranışı",
                "Microsoft Agent Framework 1.12.1 · Ollama · qwen2.5 7B",
            ],
        ),
        text=(
            "Üçüncü katman ajan. Microsoft Agent Framework üzerinde, yerelde "
            "çalışan yedi milyar parametreli bir modelle. Ajanın modele erişimi "
            "yalnızca dört araç üzerinden: kararı ve gerekçesini getiren araç, "
            "senaryo denemesi yapan araç, bir özelliğin ne olduğunu açıklayan "
            "araç ve modelin genel davranışını veren araç. Senaryo aracı önemli: "
            "tahmini uydurmuyor, modeli değiştirilmiş girdiyle gerçekten "
            "yeniden çalıştırıyor."
        ),
        speech=(
            "Üçüncü katman ajan. Maykrosoft Ejent Freymvörk üzerinde, yerelde "
            "çalışan yedi milyar parametreli bir modelle. Ajanın modele erişimi "
            "yalnızca dört araç üzerinden: kararı ve gerekçesini getiren araç, "
            "senaryo denemesi yapan araç, bir özelliğin ne olduğunu açıklayan "
            "araç ve modelin genel davranışını veren araç. Senaryo aracı önemli: "
            "tahmini uydurmuyor, modeli değiştirilmiş girdiyle gerçekten "
            "yeniden çalıştırıyor."
        ),
    ),
    Scene(
        id="d14_ui_aciklama",
        visual=ui("v_ui_explain", "explain"),
        text=(
            "Şimdi canlı arayüz. Kıl payı onaylanan başvuruyu seçiyorum ve "
            "denetimli modda açıklama üretiyorum. Şu anda yerel model araçları "
            "çağırıyor, SHAP çıktısını alıyor ve metni kuruyor. Bu tam olarak "
            "yerelde çalıştığı için sürüyor — hiçbir bulut servisine veri "
            "gitmiyor. Solda ajanın anlatısı, sağda aynı kararın SHAP şelalesi. "
            "Altta gördüğünüz satır önemli: denetçi kaç sayının temellendiğini "
            "sayıyor ve onarım devreye girdiyse söylüyor."
        ),
        speech=(
            "Şimdi canlı arayüz. Kıl payı onaylanan başvuruyu seçiyorum ve "
            "denetimli modda açıklama üretiyorum. Şu anda yerel model araçları "
            "çağırıyor, şap çıktısını alıyor ve metni kuruyor. Bu tam olarak "
            "yerelde çalıştığı için sürüyor — hiçbir bulut servisine veri "
            "gitmiyor. Solda ajanın anlatısı, sağda aynı kararın şap şelalesi. "
            "Altta gördüğünüz satır önemli: denetçi kaç sayının temellendiğini "
            "sayıyor ve onarım devreye girdiyse söylüyor."
        ),
    ),
    Scene(
        id="d15_ham",
        visual=terminal(
            "v_ham", "demo_raw",
            excerpt=("SORU: Bu başvurunun", "SORU: Kredi vadesi"), max_lines=42,
        ),
        text=(
            "Şimdi projenin can alıcı noktası: denetimi kapatıp ajanın ham "
            "çıktısına bakalım. Ekrandaki gerçek bir başarısızlık. Ajan hiç "
            "araç çağırmadı — tool çağrıları YOK yazıyor. Risk oranını doğru "
            "söylemiş, yüzde yirmi beş virgül dokuz. Ama karar eşiğini yüzde "
            "otuz diye vermiş; gerçek eşik yüzde yirmi altı. Hesap bakiyesini "
            "yüz elli mark diye uydurmuş. Kredi türü diye anlamsız bir kelime "
            "yazmış: andreavonbalken. Ve en fenası, faiz oranı diye bir "
            "özellik icat etmiş — bu veri setinde faiz oranı diye bir alan yok. "
            "Biçim kusursuz, içerik uydurma. Denetçi beş temelsiz sayıyı "
            "yakaladı: otuz, yüz elli, yedi virgül iki, beş virgül sekiz."
        ),
        speech=(
            "Şimdi projenin can alıcı noktası: denetimi kapatıp ajanın ham "
            "çıktısına bakalım. Ekrandaki gerçek bir başarısızlık. Ajan hiç "
            "araç çağırmadı — tool çağrıları YOK yazıyor. Risk oranını doğru "
            "söylemiş, yüzde yirmi beş virgül dokuz. Ama karar eşiğini yüzde "
            "otuz diye vermiş; gerçek eşik yüzde yirmi altı. Hesap bakiyesini "
            "yüz elli mark diye uydurmuş. Kredi türü diye anlamsız bir kelime "
            "yazmış: andrea von balken. Ve en fenası, faiz oranı diye bir "
            "özellik icat etmiş — bu veri setinde faiz oranı diye bir alan yok. "
            "Biçim kusursuz, içerik uydurma. Denetçi beş temelsiz sayıyı "
            "yakaladı: otuz, yüz elli, yedi virgül iki, beş virgül sekiz."
        ),
    ),
    Scene(
        id="d15b_bosluk",
        visual=card(
            "v_bosluk",
            "Denetçinin kendi sınırı",
            "Sayıyı yakaladı, terimi kaçırdı",
            [
                "✓ «%30 eşik», «150 DM», «%7.2», «%5.8» → temelsiz sayı olarak yakalandı",
                "✗ «Faiz oranı» ve «andreavonbalken» → uydurma KAVRAM listesinde yok",
                "Sayı temellendirme genel kural; kavram listesi ise elle yazılmış",
                "Bu yüzden sayı denetimi asıl iş gücü — terim denetimi yardımcı",
            ],
        ),
        text=(
            "Burada denetçinin kendi sınırını da göstermek isterim. Uydurma "
            "sayıların hepsini yakaladı, çünkü sayı denetimi genel bir kural: "
            "yükte olmayan her rakam ihlaldir. Ama faiz oranı ve "
            "andreavonbalken terimleri yakalanmadı, çünkü uydurma kavram "
            "listesi elle yazılmış sınırlı bir liste. Yani asıl iş gücü sayı "
            "denetimi; terim denetimi yardımcı. Bunu saklamak yerine söylüyorum, "
            "çünkü bir denetçinin neyi yakalamadığını bilmek, neyi yakaladığını "
            "bilmek kadar önemli."
        ),
    ),
    Scene(
        id="d16_denetimli",
        visual=terminal(
            "v_denetimli", "demo_verified",
            excerpt=("SORU: Bu başvurunun", "SORU: Kredi vadesi"), max_lines=42,
        ),
        text=(
            "Şimdi aynı soru, denetim açık. Ekranda üç şey değişti. Bir: ajan "
            "aracı çağırdı. İki: onarım devreye girdi ve üç ihlal sıfıra indi. "
            "Üç: eşik artık doğru, yüzde yirmi altı; hesap bakiyesi artık "
            "gerçek etiketiyle, sıfır ile iki yüz mark arası. On bir sayının "
            "tamamı SHAP çıktısıyla temellendirildi. Buradaki fark bir prompt "
            "iyileştirmesi değil: denetçi bağımsız bir program ve ajanın kendi "
            "hakkındaki iddialarına hiç güvenmiyor."
        ),
        speech=(
            "Şimdi aynı soru, denetim açık. Ekranda üç şey değişti. Bir: ajan "
            "aracı çağırdı. İki: onarım devreye girdi ve üç ihlal sıfıra indi. "
            "Üç: eşik artık doğru, yüzde yirmi altı; hesap bakiyesi artık "
            "gerçek etiketiyle, sıfır ile iki yüz mark arası. On bir sayının "
            "tamamı şap çıktısıyla temellendirildi. Buradaki fark bir prompt "
            "iyileştirmesi değil: denetçi bağımsız bir program ve ajanın kendi "
            "hakkındaki iddialarına hiç güvenmiyor."
        ),
    ),
    Scene(
        id="d16b_nitel",
        visual=card(
            "v_nitel",
            "Denetçinin kapsamı",
            "«Sadık» ≠ «doğru»",
            [
                "Aynı yanıtın son cümlesi: «başvurunun riski oldukça yüksektir»",
                "Oysa başvuru ONAYLANDI ve karar netliği KIL PAYI",
                "Denetim GEÇTİ — çünkü bu cümlede temelsiz sayı yok",
                "Denetçi sayısal sadakati doğrular, nitel iddiaları doğrulamaz",
            ],
        ),
        text=(
            "Ama dikkat: denetimden geçmek, doğru olmak demek değil. Aynı "
            "yanıtın son cümlesine bakın: «başvurunun riski oldukça yüksektir» "
            "diyor. Oysa bu başvuru onaylandı ve karar netliği kıl payı. Cümle "
            "yanlış, ama denetimi geçti — çünkü içinde temelsiz bir sayı yok. "
            "Denetçinin kapsamı sayısal sadakat; nitel iddiaların tutarlılığı "
            "değil. Bu, ölçtüğünüz şeyin sınırını bilmenin bir örneği: "
            "«sadakat denetimi geçti» güçlü bir garanti ama her şeyin garantisi "
            "değil."
        ),
    ),
    Scene(
        id="d17_ihlaller",
        visual=figure(
            "v_ihlaller", "08-ihlal-turleri.png",
            "Onarım döngüsünün ölçülen etkisi — ve sınırı",
        ),
        text=(
            "Denetçi yedi ihlal türü arıyor. En sık yakalananı temelsiz sayı: "
            "yükte olmayan bir rakamın metinde geçmesi. Şimdi dürüst kısım: "
            "onarım döngüsü tek soruda çok güçlü, ana açıklamada beş ihlalden "
            "sıfıra iniyor. Ama on beş yanıtın tamamına bakınca kazanç mütevazı, "
            "otuz yediden otuz ikiye. Sebep, örneklemin çoğunun takip sorusu "
            "olması. Ajan takip turunda oturum geçmişine güvenip araç çağırmıyor. "
            "Bu, prompt ile değil mimari bir kısıtla çözülecek, bilinen bir sınır."
        ),
    ),
    Scene(
        id="d18_sadakat",
        visual=figure(
            "v_sadakat", "07-sadakat-aopc.png",
            "SHAP sıralaması rastgeleden 2.16× daha etkili",
        ),
        text=(
            "Peki SHAP'ın kendisi modele sadık mı? Bunu matematiğine güvenerek "
            "değil, davranışsal olarak ölçtüm. SHAP'ın en önemli dediği "
            "özellikleri referans değerine çekip modeli yeniden çalıştırıyorum. "
            "Tahmin gerçekten düşüyorsa sıralama doğruydu. Kritik olan kontrol "
            "grubu: aynı sayıda rastgele özellik. SHAP sıralaması rastgeleden "
            "iki virgül on altı kat daha etkili. Bu karşılaştırma olmadan "
            "sıfır virgül on üçün iyi mi kötü mü olduğu bilinemezdi."
        ),
        speech=(
            "Peki şapın kendisi modele sadık mı? Bunu matematiğine güvenerek "
            "değil, davranışsal olarak ölçtüm. Şapın en önemli dediği "
            "özellikleri referans değerine çekip modeli yeniden çalıştırıyorum. "
            "Tahmin gerçekten düşüyorsa sıralama doğruydu. Kritik olan kontrol "
            "grubu: aynı sayıda rastgele özellik. Şap sıralaması rastgeleden "
            "iki virgül on altı kat daha etkili. Bu karşılaştırma olmadan "
            "sıfır virgül on üçün iyi mi kötü mü olduğu bilinemezdi."
        ),
    ),
    Scene(
        id="d19_adalet",
        visual=figure(
            "v_adalet", "09-adalet.png",
            "Korunan özellikleri çıkarmak yetmiyor",
        ),
        text=(
            "Adalet denetimi. Cinsiyet ve yabancı işçi durumunu modelden "
            "çıkardım, yine de cinsiyette yüzde on virgül üçlük bir ret oranı "
            "farkı kalıyor. Yani «bilmezlik yoluyla adalet» yetmiyor; diğer "
            "özellikler vekil görevi görüyor. Modelde bıraktığım yaş ise yüzde "
            "yirmi dokuz virgül sekizlik bir fark üretiyor. Bir kısmı gerçek "
            "risk farkından geliyor ama tamamı değil. Bu model bu haliyle "
            "üretime uygun değil ve raporda böyle yazıyor. Kleinberg ve "
            "arkadaşlarının gösterdiği gibi, bu ölçütleri aynı anda sağlamak "
            "matematiksel olarak imkânsız — hangisinin önemli olduğu bir "
            "politika kararı."
        ),
        speech=(
            "Adalet denetimi. Cinsiyet ve yabancı işçi durumunu modelden "
            "çıkardım, yine de cinsiyette yüzde on virgül üçlük bir ret oranı "
            "farkı kalıyor. Yani «bilmezlik yoluyla adalet» yetmiyor; diğer "
            "özellikler vekil görevi görüyor. Modelde bıraktığım yaş ise yüzde "
            "yirmi dokuz virgül sekizlik bir fark üretiyor. Bir kısmı gerçek "
            "risk farkından geliyor ama tamamı değil. Bu model bu haliyle "
            "üretime uygun değil ve raporda böyle yazıyor. Klaynberg ve "
            "arkadaşlarının gösterdiği gibi, bu ölçütleri aynı anda sağlamak "
            "matematiksel olarak imkânsız — hangisinin önemli olduğu bir "
            "politika kararı."
        ),
    ),
    Scene(
        id="d20_whatif",
        visual=ui("v_ui_whatif", "whatif"),
        text=(
            "Karşı-olgusal senaryo sekmesi. Kredi vadesini değiştirip modeli "
            "yeniden çalıştırıyorum. Önceki risk, yeni risk ve en çok değişen "
            "katkılar. Burada hiçbir tahmin yorumlanmıyor — LightGBM "
            "değiştirilmiş girdiyle baştan çalışıyor. Başvuru sahibinin "
            "değiştiremeyeceği özellikler ayrıca işaretli, çünkü «yaşınızı "
            "büyütün» diye bir tavsiye anlamsızdır."
        ),
        speech=(
            "Karşı olgusal senaryo sekmesi. Kredi vadesini değiştirip modeli "
            "yeniden çalıştırıyorum. Önceki risk, yeni risk ve en çok değişen "
            "katkılar. Burada hiçbir tahmin yorumlanmıyor — layt ci bi em "
            "değiştirilmiş girdiyle baştan çalışıyor. Başvuru sahibinin "
            "değiştiremeyeceği özellikler ayrıca işaretli, çünkü «yaşınızı "
            "büyütün» diye bir tavsiye anlamsızdır."
        ),
    ),
    Scene(
        id="d21_testler",
        visual=terminal("v_pytest", "pytest", max_lines=30),
        text=(
            "Kalite güvencesi. Yüz kırk sekiz test var ve önemli bir kısmı "
            "gerçek hatalardan doğdu. Denetçinin her yanlış pozitifi için bir "
            "regresyon testi yazdım: ondalık sayıları bölen cümle ayırıcı, "
            "Türkçe büyük İ harfinin küçültülmesindeki tuzak, benzer özellik "
            "adlarını karıştıran bulanık eşleştirme. Bu videoyu çekerken bir "
            "tane daha buldum: «anlamına gelir» cümlesindeki «gelir» fiilini, "
            "gelir kavramı sanıp uydurma sayıyordu. Ekranda sonda küçük bir x "
            "ve «1 xfailed» yazıyor; o, ham turda araç çağrısının kararsız "
            "olduğunu belgeleyen bilinçli bir başarısızlık beklentisi."
        ),
        speech=(
            "Kalite güvencesi. Yüz kırk sekiz test var ve önemli bir kısmı "
            "gerçek hatalardan doğdu. Denetçinin her yanlış pozitifi için bir "
            "regresyon testi yazdım: ondalık sayıları bölen cümle ayırıcı, "
            "Türkçe büyük İ harfinin küçültülmesindeki tuzak, benzer özellik "
            "adlarını karıştıran bulanık eşleştirme. Bu videoyu çekerken bir "
            "tane daha buldum: «anlamına gelir» cümlesindeki «gelir» fiilini, "
            "gelir kavramı sanıp uydurma sayıyordu. Ekranda sonda küçük bir "
            "iks ve «bir eks-feyld» yazıyor; o, ham turda araç çağrısının "
            "kararsız olduğunu belgeleyen bilinçli bir başarısızlık beklentisi."
        ),
    ),
    Scene(
        id="d22_sinirlar",
        visual=card(
            "v_sinirlar",
            "Dürüst bilanço",
            "Çözülmemiş olanlar",
            [
                "Ham ilk turda tool çağrısı kararsız — onarım döngüsü şart, süs değil",
                "Takip soruları güvenilmez: ajan oturum geçmişine güvenip sayı uyduruyor",
                "Çıktı çalıştırmaya göre oynuyor; bazı koşularda özellik adları yeniden yazılıyor",
                "15 yanıt boyunca sadakat %46.7 → %53.3 (onarımla) — tek soruda çok daha iyi",
                "Adalet: yaş uçurumu 0.298 — bu model üretime uygun DEĞİL",
            ],
        ),
        text=(
            "Kapatmadan önce çözemediklerim. Birincisi: yerel yedi milyar "
            "parametreli model ham ilk turda aracı kararsız çağırıyor. Yani "
            "onarım döngüsü bir süs değil, sistemin çalışması için zorunlu. "
            "İkincisi: takip soruları hâlâ güvenilmez. Üçüncüsü: çıktı "
            "çalıştırmadan çalıştırmaya oynuyor; bazı koşularda özellik "
            "adlarını kendi kelimeleriyle yeniden yazıyor ve o etiketleri "
            "banka kayıtlarında bulamazsınız. Dördüncüsü: adalet denetimi bu "
            "modelin üretime uygun olmadığını söylüyor. Bunları gizlemek "
            "yerine ölçüp yazdım — bir açıklanabilirlik projesinin kendisi "
            "hakkında dürüst olmaması tuhaf olurdu."
        ),
    ),
    Scene(
        id="d23_kapanis",
        visual=card(
            "v_kapanis",
            "github.com/mehmetefeaytas/xai-agent",
            "Karar modelin, gerekçe SHAP'ın, dil ajanın",
            [
                "…ve her cümle programatik olarak denetlendi.",
                "Mehmet Efe Aytaş",
            ],
        ),
        text=(
            "Özetle: kararı model veriyor, gerekçeyi SHAP hesaplıyor, dili ajan "
            "kuruyor ve ajanın her cümlesi programatik olarak denetleniyor. "
            "Kod, ölçümler ve bu videonun üretim hattı depoda açık. "
            "İzlediğiniz için teşekkürler."
        ),
        speech=(
            "Özetle: kararı model veriyor, gerekçeyi şap hesaplıyor, dili ajan "
            "kuruyor ve ajanın her cümlesi programatik olarak denetleniyor. "
            "Kod, ölçümler ve bu videonun üretim hattı depoda açık. "
            "İzlediğiniz için teşekkürler."
        ),
    ),
]


# --------------------------------------------------------------------------
# HIZLI TANITIM (~90 saniye) — görselleri derin sürümle paylaşır
# --------------------------------------------------------------------------
TEASER: list[Scene] = [
    Scene(
        id="t01_problem",
        visual=card(
            "v_teaser_baslik",
            "XAI-Agent · Mehmet Efe Aytaş",
            "Kredi kararını açıklamak",
            [
                "Model «reddedildi» der — insan «neden?» diye sorar",
                "LLM'e sormak uydurma üretir. Peki ya denetlenirse?",
            ],
        ),
        text=(
            "Bir kredi başvurusu reddedildiğinde gerekçe hakkı vardır. Ama bir "
            "dil modeline «bu kararı açıkla» derseniz, ikna edici ve yanlış bir "
            "gerekçe üretir. XAI-Agent bunu üç katmanla çözüyor."
        ),
        speech=(
            "Bir kredi başvurusu reddedildiğinde gerekçe hakkı vardır. Ama bir "
            "dil modeline «bu kararı açıkla» derseniz, ikna edici ve yanlış bir "
            "gerekçe üretir. Eks ey ay ejent bunu üç katmanla çözüyor."
        ),
    ),
    Scene(
        id="t02_selale",
        visual=figure(
            "v_selale", "05-shap-selale.png",
            "Katman 1: LightGBM karar verir · Katman 2: SHAP gerekçeyi hesaplar",
        ),
        text=(
            "LightGBM kararı verir. SHAP, kararı özelliklere ayrıştırır — ve bu "
            "ayrıştırma modelin çıktısını tam olarak yeniden kurar; sapma "
            "kayan nokta hassasiyeti kadar."
        ),
        speech=(
            "Layt ci bi em kararı verir. Şap, kararı özelliklere ayrıştırır — ve bu "
            "ayrıştırma modelin çıktısını tam olarak yeniden kurar; sapma "
            "kayan nokta hassasiyeti kadar."
        ),
    ),
    Scene(
        id="t03_ui",
        visual=ui("v_ui_explain", "explain"),
        text=(
            "Üçüncü katman, yerelde çalışan bir ajan. SHAP çıktısını dört araç "
            "üzerinden okuyup Türkçe anlatıya çeviriyor. Solda anlatı, sağda "
            "aynı kararın matematiği."
        ),
        speech=(
            "Üçüncü katman, yerelde çalışan bir ajan. Şap çıktısını dört araç "
            "üzerinden okuyup Türkçe anlatıya çeviriyor. Solda anlatı, sağda "
            "aynı kararın matematiği."
        ),
    ),
    Scene(
        id="t04_denetci",
        visual=terminal(
            "v_ham", "demo_raw",
            excerpt=("SORU: Bu başvurunun", "SORU: Kredi vadesi"), max_lines=42,
        ),
        text=(
            "Ve dördüncü katman: denetçi. Burada ajan hiç araç çağırmadan "
            "cevap verdi; karar eşiğini yanlış söyledi ve olmayan bir «faiz "
            "oranı» özelliği icat etti. Denetçi beş temelsiz sayıyı yakalayıp "
            "gerekçeleriyle modele geri verdi ve yanıt yeniden yazıldı."
        ),
    ),
    Scene(
        id="t05_kapanis",
        visual=card(
            "v_kapanis",
            "github.com/mehmetefeaytas/xai-agent",
            "Karar modelin, gerekçe SHAP'ın, dil ajanın",
            [
                "…ve her cümle programatik olarak denetlendi.",
                "Mehmet Efe Aytaş",
            ],
        ),
        text=(
            "Ölçümler, adalet denetimi ve bilinen sınırlar depoda açık. "
            "Ayrıntılı anlatım için uzun videoya bakabilirsiniz."
        ),
    ),
]


VIDEOS: dict[str, dict[str, object]] = {
    "derin": {
        "scenes": DEEP,
        "output": "xai-agent-derin-anlatim",
        "title": "XAI-Agent — derin anlatım",
    },
    "tanitim": {
        "scenes": TEASER,
        "output": "xai-agent-tanitim-90sn",
        "title": "XAI-Agent — 90 saniyelik tanıtım",
    },
}
