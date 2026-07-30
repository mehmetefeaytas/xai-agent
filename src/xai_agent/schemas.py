"""SHAP → JSON köprüsünün sözleşmesi (Pydantic şemaları).

Neden şema?
-----------
Bu dosya projenin en kritik parçası. SHAP'ın çıktısını LLM'e serbest metin
olarak verirsek ajan uydurur. Bunun yerine **katı bir sözleşme** tanımlıyoruz:
ajan yalnızca bu şemadaki alanları görür ve sistem promptu ona "bu alanların
dışına çıkma" der. Faz 5'teki faithfulness testi de tam olarak bu şemayı
referans alarak "ajan şemada olmayan bir şey söyledi mi?" sorusunu ölçer.

Alan adları Türkçe seçildi (``to_agent_payload`` içinde). Sebep: modelin
çıktı dili Türkçe olacak; anahtar adları da Türkçe olduğunda LLM'in
"çeviri yapma" yükü kalkıyor ve halüsinasyon oranı düşüyor.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Direction(StrEnum):
    """Bir özelliğin karara yönü."""

    INCREASES_RISK = "riski_artiriyor"
    DECREASES_RISK = "riski_azaltiyor"
    NEUTRAL = "etkisiz"


class ConfidenceBand(StrEnum):
    """Kararın eşiğe olan uzaklığı — "sınırda mı?" sorusunun cevabı."""

    BORDERLINE = "sinirda"
    MODERATE = "orta"
    CLEAR = "net"


#: Etki payının sözel karşılığı. 7B'lik bir modelin sayıyı yanlış yorumlama
#: riskini ortadan kaldırmak için her etkene bu etiketi de veriyoruz.
def effect_strength(share_percent: float) -> str:
    """Toplam etki içindeki payı sözel bir güç etiketine çevirir."""
    if share_percent >= 25:
        return "çok güçlü"
    if share_percent >= 15:
        return "güçlü"
    if share_percent >= 7:
        return "orta"
    return "zayıf"


_BAND_TEXT = {
    "sinirda": "KIL PAYI — karar eşiğe çok yakın, küçük bir değişiklik sonucu çevirebilir",
    "orta": "orta netlikte — eşiğe makul bir mesafe var",
    "net": "net — karar eşikten belirgin biçimde uzak",
}


class _Base(BaseModel):
    # ``model_`` ön adlı alanlar Pydantic'in korumalı ad alanıyla çakışıyor;
    # bu projede ``model_info`` gibi adlar anlamlı olduğu için korumayı kapatıyoruz.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


class FeatureContribution(_Base):
    """Tek bir özelliğin bu karara katkısı."""

    feature: str = Field(description="Makine-okunur özellik adı")
    display_name: str = Field(description="Türkçe görünen ad")
    raw_value: Any = Field(description="Ham değer (model girdisi)")
    value_label: str = Field(description="İnsan-okunur değer açıklaması")
    shap_value: float = Field(
        description="SHAP katkısı (log-odds). Pozitif = riski artırır."
    )
    abs_shap: float = Field(description="Katkının büyüklüğü")
    direction: Direction
    rank: int = Field(description="Büyüklüğe göre sıra (1 = en etkili)")
    share_of_total: float = Field(
        description="Toplam |SHAP| içindeki payı (yüzde)"
    )
    is_protected: bool = False
    is_sensitive: bool = False
    actionable: bool = False


def _pack_for_agent(c: FeatureContribution) -> dict[str, str]:
    """Bir katkıyı ajanın kopyalayabileceği saf-metin alanlara çevirir.

    Hiçbir alan sayısal tip değil: model bir sayı görürse onu yeniden
    yorumlamaya kalkıyor. Metin görürse aynen aktarıyor.
    """
    return {
        "ozellik_kodu": c.feature,
        "ad": c.display_name,
        "deger": c.value_label,
        "etki_yonu": (
            "riski artırıyor"
            if c.direction is Direction.INCREASES_RISK
            else "riski azaltıyor"
        ),
        "etki_payi": f"%{c.share_of_total:.1f}",
        "etki_gucu": effect_strength(c.share_of_total),
        "basvuru_sahibi_degistirebilir_mi": "evet" if c.actionable else "hayır",
    }


class AdditivityCheck(_Base):
    """SHAP'ın "yerel doğruluk" (local accuracy) aksiyomunun sayısal kanıtı.

    TreeSHAP'ın matematiksel garantisi şudur::

        taban_deger + Σ(tüm SHAP katkıları) = modelin ham çıktısı

    Bu eşitlik tutmuyorsa açıklama modelin gerçek davranışını temsil etmiyor
    demektir. Bu yüzden her açıklamanın içine bu kontrolü gömüyoruz — bir
    açıklama kendi doğruluğunun kanıtını taşımalı.
    """

    base_value_logodds: float
    sum_shap: float
    reconstructed_logodds: float
    model_logodds: float
    abs_error: float
    passed: bool
    tolerance: float = 1e-6


class ModelInfo(_Base):
    """Kararı üreten modelin kimlik kartı."""

    family: str
    n_features: int
    threshold: float
    positive_class_meaning: str = (
        "Pozitif sınıf = 'riskli/temerrüt'. Pozitif SHAP değeri riski ARTIRIR."
    )
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    excluded_protected_features: list[str] = Field(default_factory=list)


class DecisionExplanation(_Base):
    """Bir kredi başvurusu için tam açıklama paketi.

    Bu nesne, ajanın gördüğü **tek** bilgi kaynağıdır.
    """

    applicant_id: str | None = None
    decision: str = Field(description="'onaylandı' veya 'reddedildi'")
    risk_probability: float
    threshold: float
    margin: float = Field(description="risk_probability - threshold")
    confidence_band: ConfidenceBand

    additivity: AdditivityCheck
    top_risk_drivers: list[FeatureContribution]
    top_protective_factors: list[FeatureContribution]
    all_contributions: list[FeatureContribution]

    applicant_snapshot: dict[str, str] = Field(
        default_factory=dict,
        description="Başvurunun insan-okunur özeti (özellik -> değer açıklaması)",
    )
    model_info: ModelInfo
    fairness_note: str | None = None

    # -- yardımcılar -------------------------------------------------------
    @property
    def coverage_percent(self) -> float:
        """Gösterilen sürücülerin toplam etkinin yüzde kaçını kapsadığı."""
        shown = self.top_risk_drivers + self.top_protective_factors
        return round(sum(c.share_of_total for c in shown), 1)

    def feature_names_mentioned(self) -> set[str]:
        """Ajanın kullanmasına izin verilen özellik adları kümesi."""
        return {c.feature for c in self.all_contributions}

    def display_names_mentioned(self) -> set[str]:
        return {c.display_name for c in self.all_contributions}

    def get(self, feature: str) -> FeatureContribution | None:
        for c in self.all_contributions:
            if c.feature == feature:
                return c
        return None

    def to_agent_payload(self) -> dict[str, Any]:
        """Ajana verilecek sıkıştırılmış, Türkçe anahtarlı yük.

        Tasarım kararı: ham SHAP değeri (log-odds) buraya KONULMAZ.
        --------------------------------------------------------------
        İlk denemede 7B'lik bir modele ``katki: 0.83`` verdiğimizde model bunu
        *"%83 risk katkısı"* diye okudu. Log-odds bir yüzde değildir; bu
        doğrudan bir faithfulness ihlali.

        Çözüm: ajana hesaplaması gereken hiçbir sayı vermiyoruz. Gördüğü her
        alan **kopyalanmaya hazır bir metin**: yüzde işaretiyle birlikte pay,
        sözel etki gücü ve yön. Böylece modelin yorumlama payı sıfıra iniyor.
        Ham sayılar arayüz ve testler için :attr:`all_contributions` içinde
        korunmaya devam ediyor.
        """
        return {
            "basvuru_kimligi": self.applicant_id,
            "karar": self.decision,
            "risk_orani": f"%{self.risk_probability * 100:.1f}",
            "karar_esigi": f"%{self.threshold * 100:.1f}",
            "esik_karsilastirmasi": (
                "risk oranı eşiğin ÜSTÜNDE" if self.margin >= 0
                else "risk oranı eşiğin ALTINDA"
            ),
            "kararin_netligi": _BAND_TEXT[self.confidence_band.value],
            "riski_artiran_etkenler": [
                _pack_for_agent(c) for c in self.top_risk_drivers
            ],
            "riski_azaltan_etkenler": [
                _pack_for_agent(c) for c in self.top_protective_factors
            ],
            "gosterilen_etkenlerin_kapsami": f"%{self.coverage_percent:.0f}",
            "matematiksel_dogrulama": (
                "GEÇTİ — SHAP katkılarının toplamı modelin çıktısını birebir "
                f"yeniden kuruyor (hata {self.additivity.abs_error:.1e})"
                if self.additivity.passed
                else "BAŞARISIZ — bu açıklamaya güvenilmemeli"
            ),
            "dislanan_korunan_ozellikler": (
                self.model_info.excluded_protected_features
            ),
            "KURALLAR": [
                "Yukarıdaki 'ad', 'deger', 'etki_payi' ve 'etki_gucu' alanlarını "
                "AYNEN kullan.",
                "Burada listelenmeyen hiçbir özellikten bahsetme.",
                "Hiçbir sayıyı kendin hesaplama, dönüştürme veya yuvarlama.",
                "'etki_payi' bu etkenin toplam etki içindeki payıdır; risk "
                "olasılığı DEĞİLDİR.",
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json"), indent=indent, ensure_ascii=False
        )


class WhatIfChange(_Base):
    """What-if senaryosunda değiştirilen tek bir özellik."""

    feature: str
    display_name: str
    old_value: str
    new_value: str


class WhatIfResult(_Base):
    """Karşı-olgusal (counterfactual) senaryonun sonucu.

    Ajan "gelirim daha yüksek olsaydı ne olurdu?" sorusuna cevap verirken
    tahmin yürütmez — bu tool'u çağırır, model **gerçekten** yeniden koşar.
    Yani what-if cevapları da modele dayanıklıdır, LLM uydurması değildir.
    """

    changes: list[WhatIfChange]
    baseline_probability: float
    new_probability: float
    delta_probability: float
    baseline_decision: str
    new_decision: str
    decision_flipped: bool
    threshold: float
    top_changed_contributions: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""

    def to_agent_payload(self) -> dict[str, Any]:
        """Ajana verilecek metin-tabanlı what-if sonucu.

        Burada da ham sayı vermiyoruz: yüzde biçimlenmiş metinler ve yönü
        açıkça yazan ifadeler kullanıyoruz.
        """
        delta_pp = self.delta_probability * 100
        return {
            "yapilan_degisiklikler": [
                {
                    "ad": c.display_name,
                    "eski_deger": c.old_value,
                    "yeni_deger": c.new_value,
                }
                for c in self.changes
            ],
            "onceki_risk_orani": f"%{self.baseline_probability * 100:.1f}",
            "yeni_risk_orani": f"%{self.new_probability * 100:.1f}",
            "risk_degisimi": (
                f"{'arttı' if delta_pp > 0 else 'azaldı'} "
                f"({abs(delta_pp):.1f} puan)"
                if abs(delta_pp) >= 0.05
                else "pratikte değişmedi"
            ),
            "onceki_karar": self.baseline_decision,
            "yeni_karar": self.new_decision,
            "karar_degisti_mi": "EVET" if self.decision_flipped else "HAYIR",
            "karar_esigi": f"%{self.threshold * 100:.1f}",
            "yorum": self.note,
            "KURALLAR": [
                "Bu sonuçlar modelin GERÇEK yeniden koşumundan geldi.",
                "Yukarıdaki oranları aynen kullan; kendi tahminini ekleme.",
                "Karar değişmediyse bunu açıkça söyle, iyimser yorum yapma.",
            ],
        }


class GlobalImportanceItem(_Base):
    feature: str
    display_name: str
    mean_abs_shap: float
    percent: float
    rank: int
    direction_bias: str = Field(
        description="Bu özellik ortalamada riski artırma mı azaltma mı eğiliminde"
    )
