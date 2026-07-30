"""Faithfulness (sadakat) ölçümü — projenin bilimsel iddiasının kanıtı.

"Şeffaf yapay zeka" demek kolay. Bu modül onu **ölçüyor**.

İki bağımsız sadakat sorusu var
-------------------------------

**1. SHAP modele sadık mı?**
   SHAP güzel görünen bir açıklama üretebilir ama modelin gerçekten
   kullandığı özellikleri yansıtmıyor olabilir. Bunu ölçmek için ERASER
   literatüründen (DeYoung ve ark., 2020) iki metrik uyguluyoruz:

   * **Comprehensiveness (kapsayıcılık):** SHAP'ın "en önemli" dediği k
     özelliği referans değerlerine çekip modeli yeniden koşarız. Tahmin
     *çok* düşmeli. Düşmüyorsa SHAP yanlış özellikleri işaret ediyor.
   * **Sufficiency (yeterlilik):** Yalnızca o k özelliği bırakıp gerisini
     referansa çekeriz. Tahmin orijinaline *yakın* kalmalı. Kalmıyorsa
     açıklama eksik.

   Rastgele k özellik seçen bir kontrol grubuyla karşılaştırıyoruz — SHAP
   rastgeleyi belirgin biçimde yenmeli, yoksa sıralamanın bir değeri yok.

**2. Ajanın anlatısı SHAP'a sadık mı?**
   Bu, LLM katmanının denetimi. Ajanın ürettiği metni programatik olarak
   tarayıp şunları arıyoruz:

   * **Temellenmemiş sayı:** metinde geçip de SHAP yükünde bulunmayan sayı.
   * **Uydurulmuş özellik:** modelde olmayan bir özellikten bahsetme
     (klasik örnek: "geliriniz yetersiz" — model gelir bilgisi hiç görmez).
   * **Yön çelişkisi:** SHAP "riski azaltıyor" derken metnin "artırıyor" demesi.
   * **Korunan özellik ihlali:** cinsiyet/uyruk üzerinden gerekçe sunma.

   Prompt mühendisliği ilk savunma hattıdır ama yeterli değildir; bu modül
   ikinci hattır ve sayısal bir skor üretir.
"""

from __future__ import annotations

import contextlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .explainer import CreditExplainer
from .features import FEATURE_META
from .schemas import DecisionExplanation, Direction, WhatIfResult

# --------------------------------------------------------------------------
# Bölüm 1 — SHAP'ın modele sadakati
# --------------------------------------------------------------------------
_RNG = np.random.default_rng(config.RANDOM_SEED)


def reference_row(X_train: pd.DataFrame) -> dict[str, Any]:
    """"Özelliği kaldırmak" için kullanılacak referans (arka plan) değerleri.

    Bir özelliği gerçekten silemeyiz — model onu bekliyor. Standart çözüm
    onu "bilgi taşımayan" bir değere çekmek: sayısallarda **medyan**,
    kategoriklerde **mod** (en sık görülen değer). Böylece o özellik
    "ortalama bir başvuru gibi" davranır.
    """
    ref: dict[str, Any] = {}
    for col in X_train.columns:
        s = X_train[col]
        meta = FEATURE_META.get(col)
        if meta is not None and meta.kind == "numeric":
            ref[col] = s.median()
        else:
            mode = s.mode()
            ref[col] = mode.iloc[0] if len(mode) else s.iloc[0]
    return ref


def _mask_features(
    frame: pd.DataFrame, features: Iterable[str], ref: dict[str, Any]
) -> pd.DataFrame:
    """Belirtilen özellikleri referans değerlerine çeker."""
    out = frame.copy()
    for f in features:
        out.loc[out.index[0], f] = ref[f]
    return out


@dataclass
class ShapFaithfulnessResult:
    """SHAP açıklamalarının model sadakati ölçüm sonucu."""

    n_samples: int
    k_values: list[int]
    comprehensiveness: dict[int, float]
    sufficiency: dict[int, float]
    random_comprehensiveness: dict[int, float]
    aopc_comprehensiveness: float
    aopc_random: float
    lift_over_random: float
    additivity: dict[str, Any]

    def verdict(self) -> str:
        if not self.additivity.get("passed"):
            return "BAŞARISIZ — toplanabilirlik aksiyomu ihlal edildi"
        if self.lift_over_random < 1.2:
            return (
                "ZAYIF — SHAP sıralaması rastgele seçimden anlamlı ölçüde iyi değil"
            )
        if self.lift_over_random < 2.0:
            return "KABUL EDİLEBİLİR — SHAP rastgeleyi yeniyor"
        return "GÜÇLÜ — SHAP sıralaması modelin davranışını isabetle yakalıyor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "k_values": self.k_values,
            "comprehensiveness": {str(k): round(v, 4) for k, v in
                                  self.comprehensiveness.items()},
            "sufficiency": {str(k): round(v, 4) for k, v in
                            self.sufficiency.items()},
            "random_baseline_comprehensiveness": {
                str(k): round(v, 4) for k, v in
                self.random_comprehensiveness.items()
            },
            "aopc_comprehensiveness": round(self.aopc_comprehensiveness, 4),
            "aopc_random_baseline": round(self.aopc_random, 4),
            "lift_over_random": round(self.lift_over_random, 3),
            "additivity": self.additivity,
            "verdict": self.verdict(),
            "interpretation": {
                "comprehensiveness": (
                    "SHAP'ın en önemli dediği k özellik referansa çekildiğinde "
                    "risk tahmininin ortalama düşüşü. YÜKSEK = iyi."
                ),
                "sufficiency": (
                    "Yalnızca o k özellik bırakıldığında tahminin orijinalden "
                    "ortalama sapması. DÜŞÜK = iyi."
                ),
                "lift_over_random": (
                    "SHAP sıralamasının rastgele özellik seçimine göre kaç kat "
                    "daha etkili olduğu. 1.0 = hiç değeri yok."
                ),
            },
        }


def evaluate_shap_faithfulness(
    explainer: CreditExplainer | None = None,
    X_eval: pd.DataFrame | None = None,
    X_train: pd.DataFrame | None = None,
    k_values: Sequence[int] = (1, 2, 3, 5),
    n_samples: int = 120,
) -> ShapFaithfulnessResult:
    """SHAP açıklamalarının model sadakatini ölçer."""
    from .data import prepare

    explainer = explainer or CreditExplainer()
    if X_eval is None or X_train is None:
        ds = prepare()
        X_eval = X_eval if X_eval is not None else ds.X_test
        X_train = X_train if X_train is not None else ds.X_train

    X_eval = X_eval.head(n_samples).reset_index(drop=True)
    feats = list(explainer.bundle.feature_names)
    ref = reference_row(X_train)

    contrib, _ = explainer.contributions(X_eval)
    base_proba = explainer.bundle.predict_proba(X_eval)

    comp: dict[int, float] = {}
    suff: dict[int, float] = {}
    rand_comp: dict[int, float] = {}

    for k in k_values:
        comp_drops: list[float] = []
        suff_gaps: list[float] = []
        rand_drops: list[float] = []

        for i in range(len(X_eval)):
            row = X_eval.iloc[[i]]
            order = np.argsort(-np.abs(contrib[i]))
            top = [feats[j] for j in order[:k]]
            rest = [f for f in feats if f not in top]

            # comprehensiveness: en önemli k'yı kaldır
            p_removed = float(
                explainer.bundle.predict_proba(_mask_features(row, top, ref))[0]
            )
            comp_drops.append(base_proba[i] - p_removed)

            # sufficiency: sadece en önemli k'yı bırak
            p_only = float(
                explainer.bundle.predict_proba(_mask_features(row, rest, ref))[0]
            )
            suff_gaps.append(abs(base_proba[i] - p_only))

            # rastgele kontrol grubu
            rnd = [feats[j] for j in _RNG.choice(len(feats), size=k, replace=False)]
            p_rand = float(
                explainer.bundle.predict_proba(_mask_features(row, rnd, ref))[0]
            )
            rand_drops.append(base_proba[i] - p_rand)

        comp[int(k)] = float(np.mean(comp_drops))
        suff[int(k)] = float(np.mean(suff_gaps))
        rand_comp[int(k)] = float(np.mean(np.abs(rand_drops)))

    aopc = float(np.mean(list(comp.values())))
    aopc_rand = float(np.mean(list(rand_comp.values())))

    return ShapFaithfulnessResult(
        n_samples=len(X_eval),
        k_values=[int(k) for k in k_values],
        comprehensiveness=comp,
        sufficiency=suff,
        random_comprehensiveness=rand_comp,
        aopc_comprehensiveness=aopc,
        aopc_random=aopc_rand,
        lift_over_random=float(aopc / aopc_rand) if aopc_rand > 1e-9 else float("inf"),
        additivity=explainer.check_additivity(X_eval),
    )


# --------------------------------------------------------------------------
# Bölüm 2 — Ajan anlatısının SHAP'a sadakati
# --------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"%?\s*\d+(?:[.,]\d+)?\s*%?")

#: Modelde HİÇ olmayan ama LLM'lerin kredi bağlamında uydurmaya en meyilli
#: olduğu kavramlar. Bunlardan biri geçiyorsa halüsinasyon var.
FABRICATED_CONCEPTS = (
    "gelir",
    "maaş",
    "kredi notu",
    "kredi skoru",
    "findeks",
    "sicil",
    "icra",
    "haciz",
    "kefalet notu",
    "teminat mektubu",
    "vergi levhası",
)

#: İngilizce'ye kayma göstergeleri. Ürün Türkçe konuşan bir başvuru sahibine
#: hitap ediyor; İngilizce bir yanıt teknik olarak "sadık" olsa bile
#: kullanılamaz. Ölçümde gözlendi: onarım turundan sonra 7B model İngilizce'ye
#: kayıp tool çıktısını "Feature: ... / Impact Share: ..." diye döktü.
ENGLISH_MARKERS: tuple[str, ...] = (
    "the", "and", "for", "with", "this", "that", "from", "based",
    "impact", "decision", "threshold", "feature", "value", "share",
    "summary", "increases", "decreases", "approved", "rejected",
    "application", "outcome", "factors", "power", "direction",
)

#: Kelime sınırıyla ara. Boşlukla çevrili alt-dizi aramak yetmiyordu:
#: model "**Impact Share:**" yazdığında "impact" bir yıldıza yapışıyor ve
#: " impact " kalıbı tutmuyordu — ölçülmüş bir kaçak.
_ENGLISH_RE = re.compile(
    r"\b(?:" + "|".join(ENGLISH_MARKERS) + r")\b", re.IGNORECASE
)

#: Korunan özelliklere işaret eden ifadeler.
PROTECTED_TERMS = (
    "cinsiyet",
    "kadın",
    "erkek",
    "evli",
    "bekâr",
    "bekar",
    "medeni",
    "uyruk",
    "yabancı işçi",
    "göçmen",
)

#: Korunan bir terim geçtiğinde ihlal saymamak için aranan "dışlama" ipuçları.
#: Not: geniş zaman ("etki etmiyor") ve edilgen olumsuzluk
#: ("yola çıkılmadığı") biçimleri ölçümden sonra eklendi. Ajan doğru
#: davranıp "cinsiyetiniz bu karara etki etmiyor" dediğinde denetçi bunu
#: ihlal sayıyordu — yalnızca geçmiş zaman ("etki etmedi") tanınıyordu.
EXCLUSION_CUES = (
    "kullanmıyor",
    "kullanılmıyor",
    "kullanılmadı",
    "kullanmadı",
    "kullanmaz",
    "etki etmedi",
    "etki etmiyor",
    "etkilemiyor",
    "etkilemedi",
    "etkisi yok",
    "etkisi bulunma",
    "çıkarıl",
    "dışlan",
    "dahil edilme",
    "hesaba katılma",
    "yola çıkılma",
    "dikkate alınma",
    "görmüyor",
    "görmedi",
    "bilgi elimizde yok",
    "böyle bir bilgi yok",
)


#: "etki payı"nı bir risk artış/azalış MİKTARI gibi sunan kalıplar.
#: Gerçek bir gözlemden doğdu: ajan "%13.7 oranında riski artırıyor" yazdı.
#: Oysa %13.7 toplam etkinin içindeki paydır, riskin kendisi %44.4'tü.
#: Bu, sayının kendisi temelli olduğu hâlde ANLAMININ çarpıtılması —
#: sayı denetimi bunu yakalamaz, bu yüzden ayrı bir kontrol gerekiyor.
_MISFRAME_VERB = r"(?:artir|azalt|yukselt|dusur|getir|ekle)"
_MISFRAME_RE = re.compile(
    # A) "riski %21.3 oranında artırıyor" / "risk oranını %21.3 artırıyor"
    r"(?:risk\w*\s+(?:oranini\s+|degerini\s+)?(?:%\s*)?(?P<n1>\d+(?:[.,]\d+)?)\s*%?"
    rf"(?:\s*(?:oran|puan|kadar|olcude)\w*)?\s*{_MISFRAME_VERB}"
    # B) "%21.3 oranında artırıyor"
    r"|(?:%\s*)?(?P<n2>\d+(?:[.,]\d+)?)\s*%?\s*"
    rf"(?:oran|puan|olcude)\w*\s+{_MISFRAME_VERB}"
    # C) "%21.3 riski artırıyor" — sayı, risk kelimesinden ÖNCE.
    #    Bu alternatif ölçümden doğdu: qwen2.5 en sık bu kalıbı kuruyor ve
    #    ilk regex onu tamamen kaçırıyordu.
    r"|(?:%\s*)?(?P<n3>\d+(?:[.,]\d+)?)\s*%?\s+risk\w*"
    rf"(?:\s+\w+){{0,2}}\s+{_MISFRAME_VERB})",
    re.IGNORECASE,
)

#: Doğru çerçeveleme ipuçları — bu ifadeler varsa payı doğru sunuyor.
_CORRECT_FRAME_CUES = (
    "olustur",
    "payini",
    "payi",
    "etkenlerin",
    "kararin",
    "toplam etki",
    "agirlig",
)


_DECIMAL_GUARD = "\x00"


def _split_sentences(text: str) -> list[str]:
    """Metni cümlelere böler — ondalık sayıları PARÇALAMADAN.

    Bu fonksiyon bir hata düzeltmesinden doğdu. Başta cümleleri basitçe
    ``re.split(r"[.!?\n]+", text)`` ile bölüyorduk. Ama metinde "%21.6"
    gibi ondalık sayılar var ve nokta üzerinden bölme onları ikiye ayırıyordu:
    "riski %21" ve "6 oranında artırıyor". Sonuç olarak cümle düzeyindeki
    tüm denetimler (yön çelişkisi, korunan özellik, çerçeveleme) sessizce
    çalışmıyordu — testler "geçti" diyordu çünkü kontrol edilecek cümle
    hiç oluşmuyordu.

    Çözüm: bölmeden önce sayı içindeki ayırıcıları korumaya al, böldükten
    sonra geri koy.
    """
    guarded = re.sub(r"(?<=\d)[.,](?=\d)", _DECIMAL_GUARD, text)
    parts = re.split(r"[.!?;\n]+", guarded)
    return [p.replace(_DECIMAL_GUARD, ".").strip() for p in parts if p.strip()]


def _fold(text: str) -> str:
    """Türkçe metni karşılaştırma için normalleştirir (küçük harf, aksansız).

    Python'un ``.lower()`` metodu Türkçe'de tuzaklıdır::

        >>> repr("DEĞİLDİR".lower())
        'deği̇ldi̇r'      # 'İ' -> 'i' + U+0307 BİRLEŞİK NOKTA

    Yani küçük harfe çevirdikten sonra düz bir ``in`` karşılaştırması
    beklenmedik biçimde ``False`` döner. Bu yüzden büyük 'İ'yi ``lower()``
    çağrısından ÖNCE dönüştürüyor, ardından NFKD ile ayrıştırıp tüm birleşik
    işaretleri atıyoruz. Sonuç: 'DEĞİLDİR' ve 'değildir' aynı anahtara iner.
    """
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


#: Özellik adlarında sık geçen ve ayırt edici OLMAYAN kelimeler.
#: Bunları saymazsak "Mevcut işte çalışma **süresi**" ile "Mevcut adreste
#: ikamet **süresi**" birbirine karışıyor — ilk uygulamada tam bu oldu ve
#: denetçi 25 yanlış yön çelişkisi üretti.
_GENERIC_NAME_WORDS = frozenset(
    {
        "mevcut", "durum", "durumu", "sure", "suresi", "sayi", "sayisi",
        "oran", "orani", "kademe", "kademesi", "hesap", "hesabi", "kredi",
        "diger", "sahip", "olunan", "bankadaki", "bakmakla", "yukumlu",
        "kisi", "kayitli", "hatti", "deger", "degerli", "duzeyi", "yuku",
        "plan", "planlari", "borclu", "adreste", "adrese", "ikamet", "iste",
    }
)


def _feature_match_score(folded_sentence: str, display_name: str) -> float:
    """Cümlenin bu özellikten bahsetme gücünü puanlar.

    Ajan özellik adlarını yeniden yazıyor ("Mevcut işte çalışma süresi" ->
    "Mevcut İş Süresi"), bu yüzden tam eşleşme aramak yetmez. Ama gevşek
    eşleşme de tehlikeli: yalnızca ortak *genel* kelimelere bakan bir kontrol
    farklı özellikleri birbirine karıştırır.

    Çözüm: yalnızca **ayırt edici** kelimeleri sayıyoruz (genel kelime
    listesinde olmayan, en az 5 harfli olanlar). Tam eşleşme en yüksek puanı
    alır; böylece bir cümlede en iyi eşleşen özellik seçilebilir.
    """
    folded_name = _fold(display_name)
    if folded_name in folded_sentence:
        return 100.0

    words = [w for w in re.split(r"[^a-z0-9]+", folded_name) if w]
    distinctive = [
        w for w in words if len(w) >= 5 and w not in _GENERIC_NAME_WORDS
    ]
    if not distinctive:
        # Ayırt edici kelimesi olmayan ad (örn. "Yaş") -> tam eşleşme şart
        return 0.0

    hits = sum(1 for w in distinctive if w[: max(5, len(w) - 2)] in folded_sentence)
    required = 1 if len(distinctive) == 1 else 2
    return float(hits) if hits >= required else 0.0


def best_feature_match(folded_sentence: str, contributions: Sequence[Any]) -> Any:
    """Cümlede en güçlü biçimde anılan özelliği döndürür (yoksa ``None``).

    Bir cümle genellikle **tek** bir özellikten bahseder. Eşleşen her özelliği
    ayrı ayrı işaretlemek yerine en iyi eşleşeni seçmek yanlış pozitifleri
    büyük ölçüde kesiyor.
    """
    best, best_score = None, 0.0
    for contribution in contributions:
        score = _feature_match_score(folded_sentence, contribution.display_name)
        if score > best_score:
            best, best_score = contribution, score
    return best


#: Varsayımsal / tavsiye / eylem cümlelerinin işaretleri.
#: Bu cümleler mevcut SHAP yönü hakkında bir İDDİA değildir; "vadeyi
#: düşürerek riski azaltabilirsiniz" cümlesi vadenin şu anda riski azalttığını
#: söylemiyor. Yön ve çerçeveleme denetimleri bu cümleleri atlar.
_CONDITIONAL_CUES = (
    # kip / olasılık
    "abilir", "ebilir", "eger ", "olsa", "olursa", "duserse", "artarsa",
    "degisirse", "yaparsan", "saglayabil", "elde etme",
    # ulaç / zarf-fiil ("düşürerek", "değiştirerek")
    "duserek", "dusurerek", "artirarak", "degistirerek", "iyilestir",
    # Türkçe zaman-koşul eki: "-dığında / -duğunuzda"
    # Ölçümden doğdu: "vadeyi 12 aya düşürdüğünüzde risk azaldı" cümlesi
    # mevcut SHAP yönü hakkında bir iddia DEĞİL, bir senaryo sonucudur.
    # İlk sürüm bu cümleleri yön çelişkisi sayıyordu (8 yanlış pozitif).
    "diginde", "diginizde", "dugunda", "dugunuzde", "duginde", "duginizde",
    "tiginde", "tiginizde", "tugunda", "tugunuzde",
    # senaryo dili
    "kontrol edelim", "inceleyelim", "senaryo", "varsayim", "oneri", "tavsiye",
    "yeniden calistir", "yeniden kos",
)


def _is_conditional(folded_sentence: str) -> bool:
    return any(cue in folded_sentence for cue in _CONDITIONAL_CUES)


def mask_known_vocabulary(
    folded_text: str, explanation: DecisionExplanation
) -> str:
    """Modelin GERÇEKTEN bildiği tüm terimleri metinden siler.

    "Uydurulmuş kavram" araması için gerekli. Aksi hâlde meşru bir özellik adı
    yasaklı bir kelime içerdiğinde yanlış alarm çıkıyor: ``installment_commitment``
    özelliğinin adı **"Taksit yükü (gelire oran kademesi)"** ve içinde "gelir"
    geçiyor. Ajan bu özelliği doğru biçimde anmasına rağmen denetçi
    "gelir uydurdu" diyordu.

    Çözüm: önce modelin sözlüğünü (özellik adları + değer etiketleri) maskele,
    sonra kalan metinde yasaklı kavram ara.
    """
    masked = folded_text
    vocabulary: list[str] = []
    for c in explanation.all_contributions:
        vocabulary.append(c.display_name)
        vocabulary.append(c.value_label)
        meta = FEATURE_META.get(c.feature)
        if meta is not None:
            vocabulary.append(meta.description)
            vocabulary.extend(meta.value_labels.values())
    # Uzun ifadeleri önce sil, yoksa kısa parçalar uzunları bozar
    for term in sorted({_fold(v) for v in vocabulary if v}, key=len, reverse=True):
        if len(term) >= 4:
            masked = masked.replace(term, " ")
    return masked


def _extract_numbers(text: str) -> list[float]:
    """Metindeki tüm sayıları çıkarır."""
    out: list[float] = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace("%", "").replace(",", ".").strip()
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def allowed_numbers(
    explanation: DecisionExplanation,
    what_if_results: Sequence[WhatIfResult] = (),
) -> set[float]:
    """Ajanın kullanmasına izin verilen sayılar kümesi.

    Bu küme, ajana verilen yükte **gerçekten bulunan** her sayıyı içerir:
    risk oranı, eşik, her etkenin payı, kapsam yüzdesi, başvurudaki ham
    değerler (48 ay, 3000 DM gibi) ve what-if sonuçları. Metinde bu kümenin
    dışında bir sayı varsa ajan onu uydurmuş demektir.
    """
    allowed: set[float] = {
        round(explanation.risk_probability * 100, 1),
        round(explanation.threshold * 100, 1),
        round(abs(explanation.margin) * 100, 1),
        round(explanation.coverage_percent, 1),
        float(len(explanation.all_contributions)),
    }
    # Başvuru kimliği yükün içinde ("A-046"); ajanın onu anması meşrudur.
    # Bu, ölçülmüş bir yanlış pozitifin düzeltmesi: "Başvuru kimliği A-046"
    # cümlesindeki 46 sayısı "temellenmemiş" sayılıyordu.
    allowed.update(_extract_numbers(explanation.applicant_id or ""))
    for c in explanation.all_contributions:
        allowed.add(round(c.share_of_total, 1))
        if isinstance(c.raw_value, (int, float)) and not isinstance(c.raw_value, bool):
            allowed.add(float(c.raw_value))
        # Değer etiketinin içindeki sayılar da meşru ("100 DM altı birikim")
        allowed.update(_extract_numbers(c.value_label))

    for wif in what_if_results:
        allowed.add(round(wif.baseline_probability * 100, 1))
        allowed.add(round(wif.new_probability * 100, 1))
        allowed.add(round(abs(wif.delta_probability) * 100, 1))
        allowed.add(round(wif.threshold * 100, 1))
        for ch in wif.changes:
            allowed.update(_extract_numbers(ch.old_value))
            allowed.update(_extract_numbers(ch.new_value))

    # GENEL KURAL: ajana gösterilen yükün içinde geçen HER sayı meşrudur.
    # Bu, tek tek özel durum eklemekten (kimlik, kapsam, toplanabilirlik
    # hatası, değer etiketleri...) daha sağlam. Ölçülmüş kaçak: yük
    # "(hata 4.4e-16)" metnini taşıyor, ajan bunu aktardı ve denetçi
    # bilimsel gösterimi "4.4" ile "16" diye ikiye bölüp ikisini de
    # temellenmemiş saydı.
    with contextlib.suppress(TypeError, ValueError):
        allowed.update(
            _extract_numbers(
                json.dumps(explanation.to_agent_payload(), ensure_ascii=False)
            )
        )

    # Sıra numaraları ve küçük tam sayılar (madde numaralandırması) meşru
    allowed.update(float(i) for i in range(1, 11))
    return allowed


@dataclass
class NarrativeAudit:
    """Tek bir ajan yanıtının denetim sonucu."""

    question: str
    answer: str
    used_tools: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)
    fabricated_concepts: list[str] = field(default_factory=list)
    #: Uydurma kavramın hangi cümlede bulunduğu — teşhis ve onarım mesajı için.
    fabricated_contexts: list[dict[str, str]] = field(default_factory=list)
    direction_conflicts: list[dict[str, str]] = field(default_factory=list)
    protected_violations: list[str] = field(default_factory=list)
    misframed_shares: list[dict[str, str]] = field(default_factory=list)
    language_drift: bool = False
    missing_tool_call: bool = False
    grounded_number_count: int = 0

    @property
    def violations(self) -> int:
        return (
            len(self.ungrounded_numbers)
            + len(self.fabricated_concepts)
            + len(self.direction_conflicts)
            + len(self.protected_violations)
            + len(self.misframed_shares)
            + (1 if self.language_drift else 0)
            + (1 if self.missing_tool_call else 0)
        )

    @property
    def passed(self) -> bool:
        return self.violations == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "passed": self.passed,
            "violations": self.violations,
            "used_tools": self.used_tools,
            "grounded_number_count": self.grounded_number_count,
            "ungrounded_numbers": self.ungrounded_numbers,
            "fabricated_concepts": self.fabricated_concepts,
            "fabricated_contexts": self.fabricated_contexts,
            "direction_conflicts": self.direction_conflicts,
            "protected_violations": self.protected_violations,
            "misframed_shares": self.misframed_shares,
            "language_drift": self.language_drift,
            "missing_tool_call": self.missing_tool_call,
            "answer_excerpt": self.answer[:400],
        }


_HYPOTHETICAL_CUES = (
    "olsa",
    "olsaydi",
    "olursa",
    "dusse",
    "artsa",
    "degisse",
    "ne olur",
    "yukselse",
)


def audit_narrative(
    answer: str,
    explanation: DecisionExplanation,
    question: str = "",
    used_tools: Sequence[str] = (),
    what_if_results: Sequence[WhatIfResult] = (),
    number_tolerance: float = 0.15,
) -> NarrativeAudit:
    """Ajanın bir yanıtını SHAP gerçeğiyle karşılaştırıp ihlalleri listeler."""
    audit = NarrativeAudit(
        question=question, answer=answer, used_tools=list(used_tools)
    )
    folded = _fold(answer)
    allowed = allowed_numbers(explanation, what_if_results)
    # Kullanıcının sorusunda geçen sayılar da meşrudur: "vade 12 aya düşse"
    # diye sorulduğunda ajanın 12'yi tekrar etmesi halüsinasyon değildir.
    # (Ölçülmüş yanlış pozitif.)
    allowed.update(_extract_numbers(question))

    # 1) Temellenmemiş sayılar
    for num in _extract_numbers(answer):
        if any(abs(num - a) <= number_tolerance for a in allowed):
            audit.grounded_number_count += 1
        else:
            audit.ungrounded_numbers.append(num)

    # 2) Uydurulmuş kavramlar — modelin kendi sözlüğü maskelendikten SONRA
    masked = mask_known_vocabulary(folded, explanation)
    for concept in FABRICATED_CONCEPTS:
        folded_concept = _fold(concept)
        if folded_concept not in masked:
            continue
        audit.fabricated_concepts.append(concept)
        # Hangi cümlede geçtiğini bul — "nerede?" sorusu cevaplanabilir olmalı,
        # yoksa ne teşhis edilebilir ne de ajana anlamlı düzeltme verilebilir.
        context = ""
        for sentence in _split_sentences(answer):
            if folded_concept in mask_known_vocabulary(_fold(sentence), explanation):
                context = sentence.strip()[:160]
                break
        audit.fabricated_contexts.append(
            {"concept": concept, "sentence": context or "(cümle bulunamadı)"}
        )

    # 3) Yön çelişkileri — cümle düzeyinde, cümle başına EN İYİ eşleşen özellik
    increase_cues = ("artir", "artti", "olumsuz", "aleyhine", "yukselt")
    decrease_cues = ("azalt", "olumlu", "lehine", "dusur")
    for sentence in _split_sentences(answer):
        s = _fold(sentence)
        if not s.strip() or _is_conditional(s):
            continue  # varsayımsal/tavsiye cümlesi mevcut yön hakkında iddia değil
        says_inc = any(c in s for c in increase_cues)
        says_dec = any(c in s for c in decrease_cues)
        if says_inc == says_dec:  # ikisi de yok veya ikisi de var -> karar veremeyiz
            continue
        contribution = best_feature_match(s, explanation.all_contributions)
        if contribution is None:
            continue
        truth_inc = contribution.direction is Direction.INCREASES_RISK
        if (says_inc and not truth_inc) or (says_dec and truth_inc):
            audit.direction_conflicts.append(
                {
                    "feature": contribution.feature,
                    "shap_direction": contribution.direction.value,
                    "narrative_claim": "artırıyor" if says_inc else "azaltıyor",
                    "sentence": sentence.strip()[:160],
                }
            )

    # 4) Etki payının risk miktarı gibi sunulması (anlam çarpıtma)
    share_values = {round(c.share_of_total, 1) for c in explanation.all_contributions}
    risk_values = {
        round(explanation.risk_probability * 100, 1),
        round(explanation.threshold * 100, 1),
    }
    for wif in what_if_results:
        risk_values.update(
            {
                round(wif.baseline_probability * 100, 1),
                round(wif.new_probability * 100, 1),
                round(abs(wif.delta_probability) * 100, 1),
            }
        )
    for sentence in _split_sentences(answer):
        s = _fold(sentence)
        if (
            not s.strip()
            or any(cue in s for cue in _CORRECT_FRAME_CUES)
            or _is_conditional(s)
        ):
            continue
        for match in _MISFRAME_RE.finditer(s):
            raw = match.group("n1") or match.group("n2") or match.group("n3")
            if raw is None:
                continue
            value = float(raw.replace(",", "."))
            # Gerçek bir risk oranıysa çerçeveleme doğrudur
            if any(abs(value - r) <= number_tolerance for r in risk_values):
                continue
            if any(abs(value - sv) <= number_tolerance for sv in share_values):
                audit.misframed_shares.append(
                    {
                        "value": f"%{value}",
                        "problem": (
                            "etki payı, risk artış/azalış miktarı gibi sunuldu"
                        ),
                        "sentence": sentence.strip()[:170],
                    }
                )

    # 5) Korunan özellik ihlali
    for sentence in _split_sentences(answer):
        s = _fold(sentence)
        if not any(_fold(t) in s for t in PROTECTED_TERMS):
            continue
        if any(_fold(cue) in s for cue in EXCLUSION_CUES):
            continue  # doğru kullanım: "bu model cinsiyeti kullanmıyor"
        audit.protected_violations.append(sentence.strip()[:160])

    # 6) Dil kayması — Türkçe konuşan kullanıcıya İngilizce yanıt kullanılamaz
    english_words = {m.group(0).lower() for m in _ENGLISH_RE.finditer(folded)}
    if len(english_words) >= 4:
        audit.language_drift = True

    # 7) Varsayımsal soruda tool çağrısı zorunlu
    q = _fold(question)
    if any(cue in q for cue in _HYPOTHETICAL_CUES) and (
        "run_what_if" not in audit.used_tools
    ):
        audit.missing_tool_call = True

    return audit


@dataclass
class NarrativeFaithfulnessReport:
    """Birden çok yanıtın toplu denetim raporu."""

    audits: list[NarrativeAudit]

    @property
    def n(self) -> int:
        return len(self.audits)

    @property
    def n_passed(self) -> int:
        return sum(1 for a in self.audits if a.passed)

    @property
    def score(self) -> float:
        """Sadakat skoru: ihlalsiz yanıtların oranı."""
        return round(self.n_passed / self.n, 4) if self.n else 0.0

    @property
    def total_violations(self) -> int:
        return sum(a.violations for a in self.audits)

    def to_dict(self) -> dict[str, Any]:
        buckets = {
            "ungrounded_numbers": sum(
                len(a.ungrounded_numbers) for a in self.audits
            ),
            "fabricated_concepts": sum(
                len(a.fabricated_concepts) for a in self.audits
            ),
            "direction_conflicts": sum(
                len(a.direction_conflicts) for a in self.audits
            ),
            "protected_violations": sum(
                len(a.protected_violations) for a in self.audits
            ),
            "misframed_shares": sum(len(a.misframed_shares) for a in self.audits),
            "language_drift": sum(1 for a in self.audits if a.language_drift),
            "missing_tool_calls": sum(1 for a in self.audits if a.missing_tool_call),
        }
        return {
            "n_answers": self.n,
            "n_clean": self.n_passed,
            "faithfulness_score": self.score,
            "total_violations": self.total_violations,
            "violations_by_type": buckets,
            "verdict": self.verdict(),
            "audits": [a.to_dict() for a in self.audits],
        }

    def verdict(self) -> str:
        if self.n == 0:
            return "VERİ YOK"
        if self.score >= 0.9:
            return "GÜÇLÜ — ajan anlatısı SHAP çıktısına sadık"
        if self.score >= 0.7:
            return "KABUL EDİLEBİLİR — sınırlı sayıda ihlal var"
        return "ZAYIF — ajan yükün dışına çıkıyor, prompt sertleştirilmeli"


def save_report(payload: dict[str, Any], path=None) -> None:
    """Raporu ``artifacts/faithfulness_report.json`` dosyasına yazar."""
    config.ensure_dirs()
    (path or config.FAITHFULNESS_PATH).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
