"""Terminal demosu: üç katmanı uçtan uca gösterir (LLM isteğe bağlı).

Kullanım
--------
    uv run python scripts/explain_demo.py                 # en riskli başvuru
    uv run python scripts/explain_demo.py --index 7       # belirli başvuru
    uv run python scripts/explain_demo.py --borderline    # eşiğe en yakın başvuru
    uv run python scripts/explain_demo.py --no-llm        # yalnızca SHAP katmanı
    uv run python scripts/explain_demo.py --json          # ajan yükünü yazdır
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from xai_agent.data import prepare  # noqa: E402
from xai_agent.explainer import CreditExplainer  # noqa: E402

FOLLOW_UP_QUESTIONS = [
    "Kredi vadesi 12 aya düşse karar değişir miydi?",
    "Cinsiyetim bu karara etki etti mi?",
]


def _rule(title: str = "", char: str = "=") -> None:
    print("\n" + char * 74)
    if title:
        print(f"  {title}")
        print(char * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="XAI-Agent terminal demosu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, help="Test setindeki başvuru indeksi")
    group.add_argument(
        "--borderline",
        action="store_true",
        help="Karar eşiğine en yakın (kıl payı) başvuruyu seç",
    )
    parser.add_argument("--no-llm", action="store_true", help="Ajan katmanını atla")
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Onarım döngüsünü KAPAT. Ajanın denetlenmemiş ham çıktısını gösterir; "
            "ihlallerin ne kadar gerçek olduğunu görmek için kullanışlıdır."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Ajana giden yükü JSON olarak yazdır"
    )
    args = parser.parse_args()

    ds = prepare()
    explainer = CreditExplainer()
    bundle = explainer.bundle
    scores = bundle.predict_proba(ds.X_test)

    if args.index is not None:
        index = args.index
    elif args.borderline:
        index = int(np.argmin(np.abs(scores - bundle.threshold)))
    else:
        index = int(np.argmax(scores))

    row = ds.X_test.iloc[index].to_dict()
    applicant_id = f"A-{index:03d}"
    explanation = explainer.explain_row(row, applicant_id=applicant_id)

    # ------------------------------------------------------------------
    _rule(f"KATMAN 1 — LightGBM tahmini  ({applicant_id})")
    print(f"  Karar          : {explanation.decision.upper()}")
    print(f"  Risk oranı     : {explanation.risk_probability:.2%}")
    print(f"  Karar eşiği    : {explanation.threshold:.2%}")
    print(f"  Eşiğe uzaklık  : {explanation.margin:+.2%} "
          f"({explanation.confidence_band.value})")
    print(f"  Gerçek etiket  : "
          f"{'riskli (bad)' if ds.y_test.iloc[index] == 1 else 'iyi (good)'}")

    # ------------------------------------------------------------------
    _rule("KATMAN 2 — SHAP gerekçesi")
    a = explanation.additivity
    print(f"  Toplanabilirlik: taban {a.base_value_logodds:+.4f} "
          f"+ katkılar {a.sum_shap:+.4f} = {a.reconstructed_logodds:+.4f}")
    print(f"                   model çıktısı  = {a.model_logodds:+.4f}  "
          f"(hata {a.abs_error:.1e}) -> "
          f"{'GEÇTİ' if a.passed else 'BAŞARISIZ'}")
    print(f"\n  Riski ARTIRAN etkenler (toplam etkinin "
          f"%{sum(c.share_of_total for c in explanation.top_risk_drivers):.0f}'i):")
    for c in explanation.top_risk_drivers:
        print(f"    ▲ {c.display_name:<34} {c.value_label:<30} "
              f"%{c.share_of_total:>5.1f}  ({c.shap_value:+.4f})")
    print("\n  Riski AZALTAN etkenler:")
    for c in explanation.top_protective_factors:
        print(f"    ▼ {c.display_name:<34} {c.value_label:<30} "
              f"%{c.share_of_total:>5.1f}  ({c.shap_value:+.4f})")
    print(f"\n  Dışlanan korunan özellikler: "
          f"{explanation.model_info.excluded_protected_features}")

    if args.json:
        _rule("AJANA GİDEN YÜK (SHAP -> JSON köprüsü)")
        print(json.dumps(explanation.to_agent_payload(), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    if args.no_llm:
        _rule("KATMAN 3 atlandı (--no-llm)")
        return 0

    from xai_agent.agent import CreditAgent

    agent = CreditAgent(explainer)
    ok, msg = agent.health_check()
    _rule("KATMAN 3 — LLM ajanı")
    print(f"  LLM durumu: {msg}")
    if not ok:
        print("  -> Ajan katmanı atlanıyor. Yerel LLM'i başlatmak için: ollama serve")
        return 0

    agent.set_applicant(row, applicant_id)

    mode = "HAM (denetimsiz)" if args.raw else "DENETİMLİ (onarım döngüsü açık)"
    print(f"  Mod       : {mode}")

    questions = ["Bu başvurunun sonucunu ve nedenlerini açıkla."] + FOLLOW_UP_QUESTIONS
    for question in questions:
        _rule(f"SORU: {question}", char="-")
        if args.raw:
            turn = agent.ask_turn(question)
            audit = agent.audit_turn(turn)
            repair_info = ""
        else:
            verified = agent.ask_verified(question, max_repairs=1)
            turn, audit = verified.turn, verified.audit
            repair_info = (
                f"  [onarım devreye girdi: "
                f"{verified.first_attempt_violations} ihlal -> {audit.violations}]\n"
                if verified.was_repaired
                else ""
            )

        print(f"  [tool çağrıları: {', '.join(turn.tool_names) or 'YOK'}]")
        if repair_info:
            print(repair_info, end="")
        print()
        for line in turn.answer.splitlines():
            print(f"  {line}")
        print()
        if audit.passed:
            print(f"  ✅ SADAKAT DENETİMİ GEÇTİ "
                  f"({audit.grounded_number_count} sayı temellendirildi)")
        else:
            print(f"  ⚠️  SADAKAT DENETİMİ: {audit.violations} ihlal")
            for key, value in audit.to_dict().items():
                if key in ("question", "answer_excerpt", "passed", "violations",
                           "used_tools", "grounded_number_count"):
                    continue
                if value:
                    print(f"      {key}: {value}")

    _rule("ÖZET")
    print("  Karar LightGBM'in, gerekçe SHAP'ın, dil ajanın.")
    print("  Ajanın her cümlesi SHAP çıktısına karşı programatik olarak denetlendi.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
