"""SHAP katmanı: modelin kararını sayısal olarak parçalar.

Shapley değeri nedir?
---------------------
Oyun teorisinden gelir. Bir futbol takımı 5 gol attı; her oyuncunun bu
5 gole katkısı ne? Shapley'nin cevabı: oyuncuyu takımın **her olası alt
kümesine** ekleyip ekle-çıkar farkını ölç, sonra tüm bu farkların
ağırlıklı ortalamasını al. Adil paylaşımın matematiksel tanımı bu.

Bizim problemde "oyuncular" özellikler, "gol sayısı" ise modelin çıktısı.
SHAP değeri şu soruya cevap verir: *bu özelliğin bu değere sahip olması,
tahmini ortalamadan ne kadar uzaklaştırdı?*

Neden TreeExplainer?
--------------------
Yukarıdaki tanım naif uygulandığında 2^18 = 262.144 alt küme demek — her
başvuru için. ``KernelExplainer`` bunu rastgele örnekleyerek **yaklaşık**
hesaplar (yavaş ve gürültülü). Ağaç modelleri için Lundberg'in TreeSHAP
algoritması aynı sonucu ağaç yapısını gezerek **kesin** ve polinom zamanda
hesaplar. Yani hem hızlı hem tam doğru.

Bu projede TreeSHAP'ı LightGBM'in **kendi içine gömülü** uygulamasıyla
çağırıyoruz (``predict(pred_contrib=True)``). Sebebi:

* ``shap`` paketinin ``TreeExplainer``'ı ile birebir aynı sonucu verir
  (bkz. ``tests/test_explainer.py``, maks. fark 0.0),
* pandas ``category`` tipini doğal olarak işler — kodlamaya çevirmek gerekmez,
* ekstra bellek kopyası yaratmaz.

``shap`` paketini yalnızca grafik üretimi için kullanıyoruz.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config
from .data import align_frame
from .features import FEATURE_META, describe_value, display_name
from .model import ModelBundle
from .schemas import (
    AdditivityCheck,
    ConfidenceBand,
    DecisionExplanation,
    Direction,
    FeatureContribution,
    ModelInfo,
    WhatIfChange,
    WhatIfResult,
)

_ADDITIVITY_TOL = 1e-6


def _sigmoid(x: float | np.ndarray) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _band(margin: float) -> ConfidenceBand:
    """Kararın eşiğe uzaklığını üç kademeye ayırır.

    Bu, ajanın "kıl payı" durumları dürüstçe aktarabilmesi için var. Eşiğin
    0.01 üstünde reddedilen bir başvuruya "net biçimde riskli" demek yanıltıcı
    olur.
    """
    a = abs(margin)
    if a < 0.05:
        return ConfidenceBand.BORDERLINE
    if a < 0.15:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.CLEAR


class CreditExplainer:
    """LightGBM kararlarını SHAP ile açıklayan ana sınıf."""

    def __init__(self, bundle: ModelBundle | None = None):
        self.bundle = bundle or ModelBundle.load()
        self._booster = self.bundle.model.booster_
        self._base_value: float | None = None

    # ------------------------------------------------------------------
    # Çekirdek SHAP hesabı
    # ------------------------------------------------------------------
    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """SHAP katkılarını ve taban değerini hesaplar.

        Returns:
            ``(katkilar, taban_degerler)`` — katkılar ``(n, n_features)``
            biçiminde log-odds cinsinden, taban değerler ``(n,)``.
        """
        X = X[list(self.bundle.feature_names)]
        raw = np.asarray(self._booster.predict(X, pred_contrib=True), dtype=float)
        if raw.ndim == 1:  # tek satır bazı sürümlerde düzleşebilir
            raw = raw.reshape(1, -1)
        contrib, base = raw[:, :-1], raw[:, -1]
        if self._base_value is None:
            self._base_value = float(base[0])
        return contrib, base

    @property
    def base_value(self) -> float:
        """Modelin ortalama çıktısı (log-odds). Açıklamanın başlangıç noktası."""
        if self._base_value is None:
            self._base_value = float(self._booster.predict(
                self._reference_row(), pred_contrib=True
            )[0][-1])
        return self._base_value

    def _reference_row(self) -> pd.DataFrame:
        from .data import prepare

        return prepare().X_train.head(1)

    # ------------------------------------------------------------------
    # Toplanabilirlik (local accuracy) kontrolü
    # ------------------------------------------------------------------
    def check_additivity(self, X: pd.DataFrame) -> dict[str, Any]:
        """``taban + Σ SHAP = ham çıktı`` eşitliğini bir veri kümesinde doğrular.

        SHAP'ın en güçlü özelliği bu: açıklama, modelin çıktısını **tam olarak**
        yeniden kurar. Bu test başarısız olursa açıklamalara güvenilemez.
        """
        contrib, base = self.contributions(X)
        recon = base + contrib.sum(axis=1)
        actual = self.bundle.predict_raw(X)
        err = np.abs(recon - actual)
        return {
            "n_samples": int(len(X)),
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "tolerance": _ADDITIVITY_TOL,
            "passed": bool(err.max() < _ADDITIVITY_TOL),
            "base_value_logodds": float(base[0]),
            "base_value_probability": float(_sigmoid(base[0])),
        }

    # ------------------------------------------------------------------
    # Tek başvuru açıklaması
    # ------------------------------------------------------------------
    def explain_frame(
        self,
        frame: pd.DataFrame,
        applicant_id: str | None = None,
        top_k: int | None = None,
    ) -> DecisionExplanation:
        """Tek satırlık çerçeve için tam açıklama nesnesi üretir."""
        if len(frame) != 1:
            raise ValueError(
                f"explain_frame tek satır bekler, {len(frame)} satır geldi. "
                "Toplu açıklama için explain_batch kullanın."
            )
        top_k = top_k or config.TOP_K_DRIVERS
        feats = self.bundle.feature_names

        contrib, base = self.contributions(frame)
        shap_row = contrib[0]
        base_value = float(base[0])

        raw = float(self.bundle.predict_raw(frame)[0])
        proba = float(self.bundle.predict_proba(frame)[0])
        threshold = self.bundle.threshold
        margin = proba - threshold

        total_abs = float(np.abs(shap_row).sum()) or 1.0
        order = np.argsort(-np.abs(shap_row))

        all_contribs: list[FeatureContribution] = []
        for rank, idx in enumerate(order, start=1):
            name = feats[idx]
            meta = FEATURE_META.get(name)
            value = frame.iloc[0, list(frame.columns).index(name)]
            sv = float(shap_row[idx])
            if abs(sv) < 1e-12:
                direction = Direction.NEUTRAL
            elif sv > 0:
                direction = Direction.INCREASES_RISK
            else:
                direction = Direction.DECREASES_RISK
            all_contribs.append(
                FeatureContribution(
                    feature=name,
                    display_name=display_name(name),
                    raw_value=value.item() if hasattr(value, "item") else value,
                    value_label=describe_value(name, value),
                    shap_value=round(sv, 6),
                    abs_shap=round(abs(sv), 6),
                    direction=direction,
                    rank=rank,
                    share_of_total=round(100.0 * abs(sv) / total_abs, 3),
                    is_protected=bool(meta and meta.protected),
                    is_sensitive=bool(meta and meta.sensitive),
                    actionable=bool(meta and meta.actionable),
                )
            )

        risk_drivers = [
            c for c in all_contribs if c.direction is Direction.INCREASES_RISK
        ][:top_k]
        protective = [
            c for c in all_contribs if c.direction is Direction.DECREASES_RISK
        ][:top_k]

        metrics = self.bundle.metrics.get("test", {}) if self.bundle.metrics else {}
        dropped = (
            self.bundle.metrics.get("dataset", {}).get("dropped_protected", [])
            if self.bundle.metrics
            else []
        )

        additivity = AdditivityCheck(
            base_value_logodds=round(base_value, 8),
            sum_shap=round(float(shap_row.sum()), 8),
            reconstructed_logodds=round(base_value + float(shap_row.sum()), 8),
            model_logodds=round(raw, 8),
            abs_error=abs(base_value + float(shap_row.sum()) - raw),
            passed=abs(base_value + float(shap_row.sum()) - raw) < _ADDITIVITY_TOL,
            tolerance=_ADDITIVITY_TOL,
        )

        snapshot = {
            display_name(f): describe_value(f, frame.iloc[0][f]) for f in feats
        }

        sensitive_used = [c.display_name for c in all_contribs if c.is_sensitive]
        fairness_note = None
        if sensitive_used:
            fairness_note = (
                "Bu modelde cinsiyet ve uyruk gibi korunan özellikler kullanılmaz. "
                f"Hassas olarak izlenen özellik(ler): {', '.join(sensitive_used)}."
            )

        return DecisionExplanation(
            applicant_id=applicant_id,
            decision="reddedildi" if proba >= threshold else "onaylandı",
            risk_probability=round(proba, 6),
            threshold=round(threshold, 6),
            margin=round(margin, 6),
            confidence_band=_band(margin),
            additivity=additivity,
            top_risk_drivers=risk_drivers,
            top_protective_factors=protective,
            all_contributions=all_contribs,
            applicant_snapshot=snapshot,
            model_info=ModelInfo(
                family="LightGBM (LGBMClassifier)",
                n_features=len(feats),
                threshold=round(threshold, 6),
                test_roc_auc=metrics.get("roc_auc"),
                test_pr_auc=metrics.get("pr_auc"),
                excluded_protected_features=list(dropped),
            ),
            fairness_note=fairness_note,
        )

    def explain_row(
        self, row: dict[str, object], applicant_id: str | None = None
    ) -> DecisionExplanation:
        """Sözlük biçimindeki başvuruyu açıklar (Streamlit/what-if girişi)."""
        frame = align_frame(row, self.bundle.feature_names)
        return self.explain_frame(frame, applicant_id=applicant_id)

    def explain_batch(self, X: pd.DataFrame) -> list[DecisionExplanation]:
        """Birden çok başvuruyu açıklar (faithfulness değerlendirmesi için)."""
        return [
            self.explain_frame(X.iloc[[i]], applicant_id=f"test-{i}")
            for i in range(len(X))
        ]

    # ------------------------------------------------------------------
    # Küresel önem
    # ------------------------------------------------------------------
    def global_importance(self, X: pd.DataFrame) -> dict[str, Any]:
        """Model düzeyinde özellik önemi: ortalama |SHAP|.

        Neden ``feature_importances_`` değil? LightGBM'in yerleşik önemi
        "bu özellik kaç kez bölme yaptı" (split count) veya "ne kadar kazanç
        sağladı" (gain) der. İkisi de ölçek-bağımlı ve yanıltıcı olabilir.
        Ortalama |SHAP| ise doğrudan "çıktıyı ortalama ne kadar oynattı"
        sorusunun cevabı — birimi modelin çıktısıyla aynı (log-odds).
        """
        contrib, _ = self.contributions(X)
        mean_abs = np.abs(contrib).mean(axis=0)
        mean_signed = contrib.mean(axis=0)
        total = float(mean_abs.sum()) or 1.0
        order = np.argsort(-mean_abs)

        items: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            name = self.bundle.feature_names[idx]
            items.append(
                {
                    "feature": name,
                    "display_name": display_name(name),
                    "mean_abs_shap": round(float(mean_abs[idx]), 6),
                    "mean_signed_shap": round(float(mean_signed[idx]), 6),
                    "percent": round(100.0 * float(mean_abs[idx]) / total, 3),
                    "rank": rank,
                    "direction_bias": (
                        "ortalamada riski artırıyor"
                        if mean_signed[idx] > 0
                        else "ortalamada riski azaltıyor"
                    ),
                }
            )
        return {
            "n_samples": int(len(X)),
            "base_value_logodds": round(self.base_value, 6),
            "base_value_probability": round(float(_sigmoid(self.base_value)), 6),
            "metric": "mean(|SHAP|) — log-odds cinsinden",
            "features": items,
        }

    # ------------------------------------------------------------------
    # What-if (karşı-olgusal) analiz
    # ------------------------------------------------------------------
    def what_if(
        self,
        base_row: dict[str, object],
        changes: dict[str, object],
        top_k: int = 4,
    ) -> WhatIfResult:
        """Bazı özellikleri değiştirip modeli **gerçekten** yeniden koşar.

        Bu, ajanın "gelirim daha yüksek olsaydı?" sorusuna uydurmadan cevap
        verebilmesinin yolu. LLM tahmin yürütmez; bu fonksiyon çağrılır.
        """
        if not changes:
            raise ValueError("En az bir değişiklik belirtilmeli.")

        unknown = [k for k in changes if k not in self.bundle.feature_names]
        if unknown:
            raise ValueError(
                f"Bilinmeyen veya modelde kullanılmayan özellik(ler): {unknown}. "
                f"Geçerli özellikler: {list(self.bundle.feature_names)}"
            )

        before = self.explain_row(base_row)
        new_row = dict(base_row)
        new_row.update(changes)
        after = self.explain_row(new_row)

        change_objs = [
            WhatIfChange(
                feature=k,
                display_name=display_name(k),
                old_value=describe_value(k, base_row[k]),
                new_value=describe_value(k, v),
            )
            for k, v in changes.items()
        ]

        # Hangi katkılar en çok değişti?
        deltas: list[dict[str, Any]] = []
        for c_new in after.all_contributions:
            c_old = before.get(c_new.feature)
            if c_old is None:
                continue
            d = c_new.shap_value - c_old.shap_value
            if abs(d) < 1e-6:
                continue
            deltas.append(
                {
                    "ozellik": c_new.feature,
                    "ad": c_new.display_name,
                    "eski_katki": round(c_old.shap_value, 4),
                    "yeni_katki": round(c_new.shap_value, 4),
                    "katki_degisimi": round(d, 4),
                }
            )
        deltas.sort(key=lambda x: -abs(x["katki_degisimi"]))

        flipped = before.decision != after.decision
        note = (
            "Karar DEĞİŞTİ." if flipped
            else "Karar değişmedi; risk değeri oynadı ancak eşiğin aynı tarafında kaldı."
        )
        if any(
            not FEATURE_META[k].actionable for k in changes if k in FEATURE_META
        ):
            note += (
                " Uyarı: değiştirilen özelliklerden bazıları başvuru sahibinin kısa "
                "vadede kontrol edebileceği türden değil; bu senaryo varsayımsaldır."
            )

        return WhatIfResult(
            changes=change_objs,
            baseline_probability=round(before.risk_probability, 6),
            new_probability=round(after.risk_probability, 6),
            delta_probability=round(
                after.risk_probability - before.risk_probability, 6
            ),
            baseline_decision=before.decision,
            new_decision=after.decision,
            decision_flipped=flipped,
            threshold=round(self.bundle.threshold, 6),
            top_changed_contributions=deltas[:top_k],
            note=note,
        )

    # ------------------------------------------------------------------
    # Grafik
    # ------------------------------------------------------------------
    def waterfall_figure(
        self, explanation: DecisionExplanation, top_k: int = 10, figsize=(9, 5.5)
    ):
        """Şelale (waterfall) grafiği: taban değerden nihai karara yolculuk.

        ``shap`` paketinin hazır grafiği yerine elle çiziyoruz. Sebep:
        kategorik değerleri ve Türkçe etiketleri kontrol etmek, ayrıca
        eşik çizgisini gösterebilmek.
        """
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        contribs = sorted(
            explanation.all_contributions, key=lambda c: -c.abs_shap
        )
        shown, rest = contribs[:top_k], contribs[top_k:]
        rest_sum = sum(c.shap_value for c in rest)

        labels = [f"{c.display_name}\n({c.value_label})" for c in shown]
        values = [c.shap_value for c in shown]
        if rest:
            labels.append(f"diğer {len(rest)} özellik")
            values.append(rest_sum)

        labels, values = labels[::-1], values[::-1]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(values))
        ax.barh(y, values, color=colors, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.set_xlabel("SHAP katkısı (log-odds) — sağa = riski artırır", fontsize=9)

        for yi, v in zip(y, values):
            ax.text(
                v + (0.012 if v >= 0 else -0.012),
                yi,
                f"{v:+.3f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

        a = explanation.additivity
        ax.set_title(
            f"Karar: {explanation.decision.upper()}   |   "
            f"risk = {explanation.risk_probability:.1%}   "
            f"(eşik {explanation.threshold:.0%})\n"
            f"taban {a.base_value_logodds:+.3f} + katkılar {a.sum_shap:+.3f} "
            f"= {a.model_logodds:+.3f} log-odds",
            fontsize=10,
        )
        ax.legend(
            handles=[
                Patch(color="#d62728", label="riski artırıyor"),
                Patch(color="#2ca02c", label="riski azaltıyor"),
            ],
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )
        ax.margins(x=0.18)
        fig.tight_layout()
        return fig
