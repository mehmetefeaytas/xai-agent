"""Faz 1 + 2 çalıştırıcısı: modeli eğit, kaydet, küresel SHAP önemini üret.

Kullanım
--------
    uv run python scripts/train.py
    uv run python scripts/train.py --include-protected   # adalet karşılaştırması için
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_agent import config  # noqa: E402
from xai_agent.data import prepare  # noqa: E402
from xai_agent.model import save_metrics, train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="XAI-Agent model eğitimi")
    parser.add_argument(
        "--include-protected",
        action="store_true",
        help=(
            "Korunan özellikleri (cinsiyet/uyruk) modele DAHİL et. "
            "Sadece adalet karşılaştırması için; üretim varsayılanı değil."
        ),
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Küresel SHAP önem hesabını atla (hızlı eğitim).",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    drop_protected = not args.include_protected
    if args.include_protected:
        print(
            "\n[UYARI] Korunan özellikler modele dahil ediliyor. Bu yapılandırma "
            "yalnızca adalet karşılaştırması içindir.\n"
        )

    ds = prepare(drop_protected=drop_protected)
    bundle = train(ds)
    bundle.save()
    save_metrics(bundle)

    print(f"  Model kaydedildi   : {config.MODEL_PATH}")
    print(f"  Metrikler kaydedildi: {config.METRICS_PATH}")

    if not args.no_shap:
        from xai_agent.explainer import CreditExplainer

        print("\n  Küresel SHAP önemi hesaplanıyor...")
        expl = CreditExplainer(bundle)
        importance = expl.global_importance(ds.X_train)
        config.FEATURE_IMPORTANCE_PATH.write_text(
            json.dumps(importance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Küresel önem kaydedildi: {config.FEATURE_IMPORTANCE_PATH}")
        print("\n  En etkili 8 özellik (ortalama |SHAP|):")
        for i, item in enumerate(importance["features"][:8], 1):
            print(
                f"    {i}. {item['display_name']:<38} "
                f"{item['mean_abs_shap']:.4f}  ({item['percent']:.1f}%)"
            )

        add = expl.check_additivity(ds.X_test)
        status = "GEÇTİ" if add["passed"] else "BAŞARISIZ"
        print(
            f"\n  SHAP toplanabilirlik testi: {status} "
            f"(maks. hata {add['max_abs_error']:.2e})"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
