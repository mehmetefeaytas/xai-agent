"""Adalet (fairness) denetimi — korunan özellikleri dışlamak yeterli mi?

Ana soru
--------
Modelden cinsiyet ve uyruk bilgisini çıkardık. İş bitti mi? **Hayır.**
Buna "fairness through unawareness" (bilmezlik yoluyla adalet) denir ve
tek başına çalışmaz. Çünkü modelde kalan özellikler korunan özelliklerle
**ilişkili** olabilir: örneğin ``personal_status`` çıkarılsa bile
``housing`` veya ``job`` cinsiyetle korele olabilir ve model dolaylı olarak
aynı ayrımı öğrenebilir. Buna *vekil ayrımcılık* (proxy discrimination) denir.

Bu modül üç standart metriği hesaplar:

**Demographic parity (demografik eşitlik)**
    Gruplar arasında red oranı farkı. ``P(red | grup A) - P(red | grup B)``.
    Sıfıra yakın olmalı.

**Equal opportunity (fırsat eşitliği)**
    Gerçekten riskli olanlar arasında yakalama oranı (recall) farkı.
    Hardt ve ark. (2016). Sıfıra yakın olmalı.

**Predictive parity (öngörü eşitliği)**
    Reddedilenler arasında gerçekten riskli olma oranı (precision) farkı.

Not: bu üç metriğin aynı anda sıfırlanması matematiksel olarak genellikle
**imkânsızdır** (Kleinberg ve ark., 2016 — adalet metriklerinin bağdaşmazlığı).
Bu yüzden rapor bir "geçti/kaldı" damgası vermez; farkları ölçüp karar
vericinin önüne koyar. Dürüst olan yaklaşım budur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_META
from .model import ModelBundle

#: Bir grubun metriklerinin anlamlı sayılması için gereken en az örnek sayısı.
MIN_GROUP_SIZE = 20


@dataclass
class GroupMetrics:
    """Tek bir demografik grubun sonuç metrikleri."""

    group: str
    label: str
    n: int
    rejection_rate: float
    actual_bad_rate: float
    recall: float | None
    precision: float | None
    mean_risk_score: float
    reliable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "label": self.label,
            "n": self.n,
            "rejection_rate": round(self.rejection_rate, 4),
            "actual_bad_rate": round(self.actual_bad_rate, 4),
            "recall": None if self.recall is None else round(self.recall, 4),
            "precision": None if self.precision is None else round(self.precision, 4),
            "mean_risk_score": round(self.mean_risk_score, 4),
            "reliable": self.reliable,
            "note": (
                ""
                if self.reliable
                else f"Grup {self.n} kişiden oluşuyor (<{MIN_GROUP_SIZE}); "
                "metrikler istatistiksel olarak güvenilmez."
            ),
        }


def _group_metrics(
    name: str,
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> GroupMetrics:
    n = int(len(y_true))
    positives = y_true == 1
    predicted_pos = y_pred == 1
    return GroupMetrics(
        group=name,
        label=label,
        n=n,
        rejection_rate=float(y_pred.mean()) if n else 0.0,
        actual_bad_rate=float(y_true.mean()) if n else 0.0,
        recall=(
            float(y_pred[positives].mean()) if positives.sum() > 0 else None
        ),
        precision=(
            float(y_true[predicted_pos].mean()) if predicted_pos.sum() > 0 else None
        ),
        mean_risk_score=float(y_score.mean()) if n else 0.0,
        reliable=n >= MIN_GROUP_SIZE,
    )


def _gap(values: list[float | None]) -> float | None:
    """Bir metriğin gruplar arası en büyük farkı."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(max(clean) - min(clean))


def audit_attribute(
    attribute: str,
    groups: pd.Series,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Tek bir korunan özellik için grup metriklerini ve farkları hesaplar."""
    y_true = np.asarray(y_true)
    meta = FEATURE_META.get(attribute)
    labels = meta.value_labels if meta else {}

    per_group: list[GroupMetrics] = []
    for value in sorted(pd.Series(groups).astype(str).unique()):
        mask = np.asarray(pd.Series(groups).astype(str) == value)
        if mask.sum() == 0:
            continue
        per_group.append(
            _group_metrics(
                name=value,
                label=labels.get(value, value),
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                y_score=y_score[mask],
            )
        )

    reliable = [g for g in per_group if g.reliable]
    return {
        "attribute": attribute,
        "display_name": meta.display if meta else attribute,
        "in_model": not (meta and meta.protected),
        "groups": [g.to_dict() for g in per_group],
        "gaps": {
            "demographic_parity_gap": _gap([g.rejection_rate for g in reliable]),
            "equal_opportunity_gap": _gap([g.recall for g in reliable]),
            "predictive_parity_gap": _gap([g.precision for g in reliable]),
            "mean_score_gap": _gap([g.mean_risk_score for g in reliable]),
            "actual_bad_rate_gap": _gap([g.actual_bad_rate for g in reliable]),
        },
        "n_reliable_groups": len(reliable),
    }


def interpret_gap(gap: float | None) -> str:
    """Bir fark değerini sözel bir yoruma çevirir."""
    if gap is None:
        return "hesaplanamadı (yetersiz grup verisi)"
    if gap < 0.05:
        return "ihmal edilebilir (<5 puan)"
    if gap < 0.10:
        return "küçük (5–10 puan)"
        # 10 puanın üzeri kredi bağlamında incelemeyi hak eder
    if gap < 0.20:
        return "DİKKAT — belirgin fark (10–20 puan), incelenmeli"
    return "CİDDİ — büyük fark (>20 puan), model kullanılmamalı"


def run_fairness_audit(
    bundle: ModelBundle | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Test seti üzerinde tam adalet denetimi yürütür.

    Modelde bulunan hassas özellikler (``age``) ve modelden çıkarılmış korunan
    özellikler (``personal_status``, ``foreign_worker``) için ayrı ayrı
    ölçüm yapar. İkincisi asıl ilginç olan: özellik modelde olmadığı hâlde
    gruplar arasında fark çıkıyorsa **vekil ayrımcılık** var demektir.
    """
    from .data import prepare

    bundle = bundle or ModelBundle.load()
    ds = prepare()

    y_score = bundle.predict_proba(ds.X_test)
    y_pred = (y_score >= bundle.threshold).astype(int)
    y_true = np.asarray(ds.y_test)

    audits: list[dict[str, Any]] = []

    # 1) Modelden çıkarılmış korunan özellikler -> vekil ayrımcılık testi
    for attribute in ds.protected_test.columns:
        audits.append(
            audit_attribute(
                attribute, ds.protected_test[attribute], y_true, y_pred, y_score
            )
        )

    # 2) Modelde bulunan hassas özellik: yaş -> gruplandırarak incele
    if "age" in ds.X_test.columns:
        age_bins = pd.cut(
            ds.X_test["age"],
            bins=[18, 30, 40, 55, 100],
            labels=["19-30", "31-40", "41-55", "56+"],
        ).astype(str)
        audit = audit_attribute("age", age_bins, y_true, y_pred, y_score)
        audit["in_model"] = True
        audit["display_name"] = "Yaş grubu"
        audits.append(audit)

    summary = []
    for audit in audits:
        dp = audit["gaps"]["demographic_parity_gap"]
        eo = audit["gaps"]["equal_opportunity_gap"]
        summary.append(
            {
                "attribute": audit["display_name"],
                "in_model": audit["in_model"],
                "demographic_parity_gap": (
                    None if dp is None else round(dp, 4)
                ),
                "demographic_parity_verdict": interpret_gap(dp),
                "equal_opportunity_gap": (
                    None if eo is None else round(eo, 4)
                ),
                "equal_opportunity_verdict": interpret_gap(eo),
            }
        )

    report = {
        "n_test": int(len(ds.X_test)),
        "threshold": round(bundle.threshold, 4),
        "excluded_from_model": list(ds.protected_test.columns),
        "method_note": (
            "Korunan özellikler model girdisinden çıkarıldı ('fairness through "
            "unawareness'). Bu tek başına yetmez: aşağıdaki farklar, modelde "
            "kalan özellikler üzerinden VEKİL AYRIMCILIK olup olmadığını gösterir."
        ),
        "impossibility_note": (
            "Demografik eşitlik, fırsat eşitliği ve öngörü eşitliğinin aynı anda "
            "sağlanması taban oranlar eşit değilse matematiksel olarak imkânsızdır "
            "(Kleinberg ve ark., 2016). Bu rapor geçti/kaldı damgası vermez; "
            "farkları ölçüp karar vericiye sunar."
        ),
        "summary": summary,
        "details": audits,
    }

    if save:
        config.ensure_dirs()
        config.FAIRNESS_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def print_fairness_report(report: dict[str, Any]) -> None:
    """Adalet raporunu terminale okunur biçimde basar."""
    print("\n" + "=" * 76)
    print("  ADALET DENETİMİ — test seti")
    print("=" * 76)
    print(f"  Modelden çıkarılan: {report['excluded_from_model']}")
    print(f"  Karar eşiği       : {report['threshold']}")
    print("-" * 76)
    for item in report["summary"]:
        flag = "modelde" if item["in_model"] else "DIŞLANMIŞ"
        print(f"\n  {item['attribute']}  [{flag}]")
        dp, eo = item["demographic_parity_gap"], item["equal_opportunity_gap"]
        print(
            f"    Red oranı farkı      : "
            f"{'—' if dp is None else f'{dp:.3f}'}  → "
            f"{item['demographic_parity_verdict']}"
        )
        print(
            f"    Fırsat eşitliği farkı: "
            f"{'—' if eo is None else f'{eo:.3f}'}  → "
            f"{item['equal_opportunity_verdict']}"
        )
    print("\n" + "-" * 76)
    print("  " + report["impossibility_note"].replace(". ", ".\n  "))
    print("=" * 76 + "\n")
