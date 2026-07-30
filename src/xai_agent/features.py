"""Özellik sözlüğü: makine adlarını insan diline çeviren katman.

Neden bu dosya var?
-------------------
SHAP bize ``checking_status = '<0'`` için ``+0.42`` gibi bir sayı verir.
Bu, LLM'e ham hâlde verilirse ajan uydurmaya başlar ("negatif bakiye"nin ne
olduğunu tahmin etmeye çalışır). Bunun yerine her özelliğin ve her kategori
değerinin Türkçe karşılığını burada **sabitliyoruz**. Ajan böylece
yorumlamak zorunda kalmaz, sadece aktarır.

Ayrıca kategori seviyelerini burada sabitlemek kritik bir teknik gerekliliktir:
LightGBM kategorik özellikleri pandas'ın ``category`` kodlarına göre öğrenir.
Eğitim ve çıkarım sırasında kategori sırası farklı olursa model sessizce
yanlış tahmin yapar. Bu sözlük o sırayı tek yerde sabitler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureMeta:
    """Tek bir özelliğin insan-okunur tanımı."""

    name: str
    display: str
    kind: FeatureKind
    description: str
    unit: str | None = None
    #: Kategorik özellikler için sabit seviye sırası (LightGBM kodlaması bunu kullanır).
    levels: tuple[str, ...] = ()
    #: Ham kategori değeri -> Türkçe açıklama.
    value_labels: dict[str, str] = field(default_factory=dict)
    #: Numerik özellikler için (min, max, adım) — what-if arayüzü bunu kullanır.
    bounds: tuple[float, float, float] | None = None
    #: Yasal/etik olarak kredi kararında kullanılmaması gereken özellik mi?
    protected: bool = False
    #: Modelde tutulan ama adalet denetiminde izlenen özellik mi?
    sensitive: bool = False
    #: Başvuru sahibinin kısa vadede değiştirebileceği bir özellik mi?
    #: What-if önerileri yalnızca bunların üzerinden yapılır.
    actionable: bool = False

    def label_for(self, value: Any) -> str:
        """Ham değeri insan-okunur metne çevirir."""
        if self.kind == "numeric":
            unit = f" {self.unit}" if self.unit else ""
            if isinstance(value, float) and not value.is_integer():
                return f"{value:.2f}{unit}"
            return f"{int(value)}{unit}"
        return self.value_labels.get(str(value), str(value))


def _f(**kwargs: Any) -> FeatureMeta:
    return FeatureMeta(**kwargs)


FEATURE_META: dict[str, FeatureMeta] = {
    "checking_status": _f(
        name="checking_status",
        display="Vadesiz hesap durumu",
        kind="categorical",
        description=(
            "Başvuru sahibinin bankadaki vadesiz (cari) hesabındaki bakiye aralığı. "
            "Kredi riski tahmininde en güçlü tek sinyaldir."
        ),
        levels=("<0", "0<=X<200", ">=200", "no checking"),
        value_labels={
            "<0": "bakiye negatif (0 DM altı)",
            "0<=X<200": "bakiye 0–200 DM arası",
            ">=200": "bakiye 200 DM ve üzeri",
            "no checking": "vadesiz hesabı yok",
        },
        actionable=True,
    ),
    "duration": _f(
        name="duration",
        display="Kredi vadesi",
        kind="numeric",
        unit="ay",
        description="Kredinin geri ödeme süresi. Uzun vade genellikle riski artırır.",
        bounds=(4, 72, 1),
        actionable=True,
    ),
    "credit_history": _f(
        name="credit_history",
        display="Kredi geçmişi",
        kind="categorical",
        description="Başvuru sahibinin geçmiş kredi ödeme davranışı.",
        levels=(
            "no credits/all paid",
            "all paid",
            "existing paid",
            "delayed previously",
            "critical/other existing credit",
        ),
        value_labels={
            "no credits/all paid": "hiç kredisi olmamış veya tümü kapatılmış",
            "all paid": "bu bankadaki tüm krediler zamanında ödendi",
            "existing paid": "mevcut krediler düzenli ödeniyor",
            "delayed previously": "geçmişte ödeme gecikmesi yaşandı",
            "critical/other existing credit": (
                "kritik hesap veya başka bankada devam eden kredi var"
            ),
        },
    ),
    "purpose": _f(
        name="purpose",
        display="Kredi kullanım amacı",
        kind="categorical",
        description="Kredinin hangi harcama için talep edildiği.",
        levels=(
            "new car",
            "used car",
            "furniture/equipment",
            "radio/tv",
            "domestic appliance",
            "repairs",
            "education",
            "retraining",
            "business",
            "other",
        ),
        value_labels={
            "new car": "sıfır araç alımı",
            "used car": "ikinci el araç alımı",
            "furniture/equipment": "mobilya / ev eşyası",
            "radio/tv": "elektronik (radyo, televizyon)",
            "domestic appliance": "beyaz eşya",
            "repairs": "onarım / tadilat",
            "education": "eğitim",
            "retraining": "mesleki yeniden eğitim",
            "business": "işletme / ticari amaç",
            "other": "diğer",
        },
        actionable=True,
    ),
    "credit_amount": _f(
        name="credit_amount",
        display="Kredi tutarı",
        kind="numeric",
        unit="DM",
        description="Talep edilen kredi miktarı (Alman Markı).",
        bounds=(250, 18424, 50),
        actionable=True,
    ),
    "savings_status": _f(
        name="savings_status",
        display="Tasarruf hesabı durumu",
        kind="categorical",
        description="Başvuru sahibinin tasarruf/birikim hesabındaki tutar aralığı.",
        levels=("<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"),
        value_labels={
            "<100": "100 DM altı birikim",
            "100<=X<500": "100–500 DM birikim",
            "500<=X<1000": "500–1000 DM birikim",
            ">=1000": "1000 DM ve üzeri birikim",
            "no known savings": "bilinen bir birikimi yok",
        },
        actionable=True,
    ),
    "employment": _f(
        name="employment",
        display="Mevcut işte çalışma süresi",
        kind="categorical",
        description="Başvuru sahibinin şu anki işinde ne kadar süredir çalıştığı.",
        levels=("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"),
        value_labels={
            "unemployed": "işsiz",
            "<1": "1 yıldan az",
            "1<=X<4": "1–4 yıl",
            "4<=X<7": "4–7 yıl",
            ">=7": "7 yıl ve üzeri",
        },
    ),
    "installment_commitment": _f(
        name="installment_commitment",
        display="Taksit yükü (gelire oran kademesi)",
        kind="numeric",
        description=(
            "Aylık taksitin harcanabilir gelire oranını gösteren 1–4 arası kademe. "
            "4 en yüksek taksit yükünü ifade eder."
        ),
        bounds=(1, 4, 1),
        actionable=True,
    ),
    "personal_status": _f(
        name="personal_status",
        display="Medeni durum ve cinsiyet",
        kind="categorical",
        description=(
            "KORUNAN ÖZELLİK. Cinsiyet ve medeni durum bilgisi içerir. Kredi "
            "kararında kullanılması ayrımcılık oluşturur; bu projede model "
            "eğitiminden dışlanır ve yalnızca adalet denetiminde kullanılır."
        ),
        levels=("male single", "male mar/wid", "male div/sep", "female div/dep/mar"),
        value_labels={
            "male single": "erkek, bekâr",
            "male mar/wid": "erkek, evli veya dul",
            "male div/sep": "erkek, boşanmış veya ayrı",
            "female div/dep/mar": "kadın (boşanmış / ayrı / evli)",
        },
        protected=True,
    ),
    "other_parties": _f(
        name="other_parties",
        display="Diğer borçlu veya kefil",
        kind="categorical",
        description="Kredide müşterek borçlu ya da kefil bulunup bulunmadığı.",
        levels=("none", "co applicant", "guarantor"),
        value_labels={
            "none": "yok",
            "co applicant": "müşterek borçlu var",
            "guarantor": "kefil var",
        },
        actionable=True,
    ),
    "residence_since": _f(
        name="residence_since",
        display="Mevcut adreste ikamet süresi (kademe)",
        kind="numeric",
        description="Aynı adreste oturma süresini gösteren 1–4 arası kademe.",
        bounds=(1, 4, 1),
    ),
    "property_magnitude": _f(
        name="property_magnitude",
        display="Sahip olunan en değerli varlık",
        kind="categorical",
        description="Başvuru sahibinin teminat gösterebileceği en değerli varlık türü.",
        levels=("real estate", "life insurance", "car", "no known property"),
        value_labels={
            "real estate": "gayrimenkul",
            "life insurance": "hayat sigortası veya birikim poliçesi",
            "car": "araç veya benzeri varlık",
            "no known property": "bilinen bir varlığı yok",
        },
    ),
    "age": _f(
        name="age",
        display="Yaş",
        kind="numeric",
        unit="yaş",
        description=(
            "Başvuru sahibinin yaşı. Hassas özellik olarak işaretlidir: modelde "
            "kullanılır ancak adalet denetiminde grup farkları izlenir."
        ),
        bounds=(19, 75, 1),
        sensitive=True,
    ),
    "other_payment_plans": _f(
        name="other_payment_plans",
        display="Diğer taksit planları",
        kind="categorical",
        description="Başka banka veya mağazalarda devam eden taksit yükümlülükleri.",
        levels=("none", "bank", "stores"),
        value_labels={
            "none": "yok",
            "bank": "başka bankada taksitli borç var",
            "stores": "mağaza taksiti var",
        },
        actionable=True,
    ),
    "housing": _f(
        name="housing",
        display="Konut durumu",
        kind="categorical",
        description="Başvuru sahibinin oturduğu evin mülkiyet durumu.",
        levels=("own", "rent", "for free"),
        value_labels={
            "own": "ev sahibi",
            "rent": "kirada oturuyor",
            "for free": "ücretsiz oturuyor (aile yanı vb.)",
        },
    ),
    "existing_credits": _f(
        name="existing_credits",
        display="Bu bankadaki mevcut kredi sayısı",
        kind="numeric",
        unit="adet",
        description="Başvuru sahibinin aynı bankada devam eden kredi sayısı.",
        bounds=(1, 4, 1),
        actionable=True,
    ),
    "job": _f(
        name="job",
        display="Meslek ve nitelik düzeyi",
        kind="categorical",
        description="Başvuru sahibinin mesleki nitelik kategorisi.",
        levels=(
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt",
        ),
        value_labels={
            "unemp/unskilled non res": "işsiz veya vasıfsız (yerleşik değil)",
            "unskilled resident": "vasıfsız çalışan (yerleşik)",
            "skilled": "vasıflı çalışan veya memur",
            "high qualif/self emp/mgmt": "yüksek nitelikli, yönetici veya serbest meslek",
        },
    ),
    "num_dependents": _f(
        name="num_dependents",
        display="Bakmakla yükümlü kişi sayısı",
        kind="numeric",
        unit="kişi",
        description="Başvuru sahibinin geçimini sağladığı kişi sayısı.",
        bounds=(1, 2, 1),
    ),
    "own_telephone": _f(
        name="own_telephone",
        display="Kayıtlı telefon hattı",
        kind="categorical",
        description="Başvuru sahibi adına kayıtlı sabit telefon bulunup bulunmadığı.",
        levels=("none", "yes"),
        value_labels={"none": "yok", "yes": "var"},
        actionable=True,
    ),
    "foreign_worker": _f(
        name="foreign_worker",
        display="Yabancı işçi durumu",
        kind="categorical",
        description=(
            "KORUNAN ÖZELLİK. Uyruk/göçmenlik durumunu ifade eder. Kredi kararında "
            "kullanılması ayrımcılık oluşturur; model eğitiminden dışlanır."
        ),
        levels=("no", "yes"),
        value_labels={"no": "hayır", "yes": "evet"},
        protected=True,
    ),
}

#: Veri setindeki tüm özellik adları (hedef hariç), veri setindeki sırayla.
ALL_FEATURES: tuple[str, ...] = tuple(FEATURE_META.keys())

CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    n for n, m in FEATURE_META.items() if m.kind == "categorical"
)
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    n for n, m in FEATURE_META.items() if m.kind == "numeric"
)
PROTECTED_FEATURE_NAMES: tuple[str, ...] = tuple(
    n for n, m in FEATURE_META.items() if m.protected
)
ACTIONABLE_FEATURES: tuple[str, ...] = tuple(
    n for n, m in FEATURE_META.items() if m.actionable
)

#: Kategorik özellik -> sabit seviye sırası. LightGBM kodlaması bunu kullanır.
CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    n: FEATURE_META[n].levels for n in CATEGORICAL_FEATURES
}


def get_meta(name: str) -> FeatureMeta:
    """Özellik adından meta bilgisini döndürür.

    Bilinmeyen bir ad gelirse ``KeyError`` yerine anlamlı bir hata veririz —
    ajanın uydurduğu bir özellik adını buradan yakalayabiliyoruz.
    """
    try:
        return FEATURE_META[name]
    except KeyError as exc:  # pragma: no cover - savunma amaçlı
        raise KeyError(
            f"'{name}' bilinen bir özellik değil. Geçerli özellikler: "
            f"{', '.join(ALL_FEATURES)}"
        ) from exc


def display_name(name: str) -> str:
    """Özelliğin Türkçe görünen adı (bilinmeyen ad ham hâlde döner)."""
    meta = FEATURE_META.get(name)
    return meta.display if meta else name


def describe_value(name: str, value: object) -> str:
    """``feature=value`` çiftini insan-okunur metne çevirir."""
    meta = FEATURE_META.get(name)
    if meta is None:
        return f"{name} = {value}"
    return meta.label_for(value)


def model_features(drop_protected: bool = True) -> tuple[str, ...]:
    """Modelin eğitiminde kullanılacak özellik listesi.

    ``drop_protected=True`` iken cinsiyet/uyruk gibi korunan özellikler
    dışlanır. Bu, projenin etik varsayılanıdır.
    """
    if not drop_protected:
        return ALL_FEATURES
    return tuple(n for n in ALL_FEATURES if not FEATURE_META[n].protected)


def feature_catalog(drop_protected: bool = True) -> list[dict[str, object]]:
    """Ajanın ``get_feature_info`` tool'u için makine-okunur özellik kataloğu."""
    names = model_features(drop_protected=drop_protected)
    catalog: list[dict[str, object]] = []
    for name in names:
        m = FEATURE_META[name]
        entry: dict[str, object] = {
            "feature": name,
            "display_name": m.display,
            "kind": m.kind,
            "description": m.description,
            "actionable": m.actionable,
        }
        if m.kind == "categorical":
            entry["allowed_values"] = [
                {"value": lvl, "label": m.value_labels.get(lvl, lvl)} for lvl in m.levels
            ]
        else:
            if m.bounds:
                lo, hi, step = m.bounds
                entry["range"] = {"min": lo, "max": hi, "step": step}
            if m.unit:
                entry["unit"] = m.unit
        catalog.append(entry)
    return catalog
