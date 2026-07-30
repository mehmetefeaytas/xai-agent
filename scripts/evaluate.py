"""Faz 5 çalıştırıcısı: sadakat (faithfulness) + adalet (fairness) denetimi.

Kullanım
--------
    uv run python scripts/evaluate.py                 # tam denetim
    uv run python scripts/evaluate.py --skip-llm      # LLM'siz (hızlı, offline)
    uv run python scripts/evaluate.py --n 8           # 8 başvuru üzerinde ajan denetimi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from xai_agent import config  # noqa: E402
from xai_agent.data import prepare  # noqa: E402
from xai_agent.explainer import CreditExplainer  # noqa: E402
from xai_agent.fairness import print_fairness_report, run_fairness_audit  # noqa: E402
from xai_agent.faithfulness import (  # noqa: E402
    NarrativeFaithfulnessReport,
    audit_narrative,
    evaluate_shap_faithfulness,
    save_report,
)
from xai_agent.tools import what_ifs_from_calls  # noqa: E402

#: Ajanın denetlendiği soru seti. Üçüncüsü kasıtlı bir tuzak: varsayımsal
#: soruda tool çağırmadan cevap verirse ihlal sayılır.
AUDIT_QUESTIONS = [
    "Bu başvurunun sonucunu ve nedenlerini başvuru sahibine açıkla.",
    "Bu kararda lehime olan etkenler nelerdi?",
    "Kredi vadesi 12 aya düşse karar değişir miydi?",
]


def run_shap_section(n_samples: int) -> dict:
    print("\n" + "=" * 76)
    print("  BÖLÜM 1 — SHAP MODELE SADIK MI?  (LLM gerekmez)")
    print("=" * 76)
    explainer = CreditExplainer()
    ds = prepare()
    result = evaluate_shap_faithfulness(
        explainer, X_eval=ds.X_test, X_train=ds.X_train, n_samples=n_samples
    )
    d = result.to_dict()

    add = d["additivity"]
    print(
        f"  Toplanabilirlik (yerel doğruluk): "
        f"{'GEÇTİ' if add['passed'] else 'BAŞARISIZ'} "
        f"(maks. hata {add['max_abs_error']:.2e})"
    )
    print(f"  Değerlendirilen başvuru sayısı  : {d['n_samples']}")
    print("-" * 76)
    print(f"  {'k':>3}  {'Comprehensiveness':>19}  {'Sufficiency':>13}  {'Rastgele':>10}")
    print(f"  {'':>3}  {'(yüksek = iyi)':>19}  {'(düşük = iyi)':>13}  {'kontrol':>10}")
    print("-" * 76)
    for k in d["k_values"]:
        print(
            f"  {k:>3}  {d['comprehensiveness'][str(k)]:>19.4f}  "
            f"{d['sufficiency'][str(k)]:>13.4f}  "
            f"{d['random_baseline_comprehensiveness'][str(k)]:>10.4f}"
        )
    print("-" * 76)
    print(
        f"  AOPC (SHAP)      : {d['aopc_comprehensiveness']:.4f}\n"
        f"  AOPC (rastgele)  : {d['aopc_random_baseline']:.4f}\n"
        f"  SHAP kazancı     : {d['lift_over_random']:.2f}x\n"
        f"  KARAR            : {d['verdict']}"
    )
    return d


def run_narrative_section(n_applicants: int, repair_rounds: int = 0) -> dict | None:
    print("\n" + "=" * 76)
    print("  BÖLÜM 2 — AJAN ANLATISI SHAP'A SADIK MI?  (LLM gerekir)")
    print("=" * 76)

    from xai_agent.agent import CreditAgent

    agent = CreditAgent()
    ok, msg = agent.health_check()
    print(f"  LLM durumu: {msg}")
    if not ok:
        print("  -> LLM kullanılamıyor, bu bölüm atlanıyor.")
        return None

    ds = prepare()
    scores = agent.explainer.bundle.predict_proba(ds.X_test)
    # Uçlardan ve sınırdan başvuru seç: en riskli, en güvenli ve eşiğe en yakın
    order = np.argsort(scores)
    borderline = np.argsort(np.abs(scores - agent.explainer.bundle.threshold))
    picks: list[int] = []
    for idx in list(order[::-1]) + list(borderline) + list(order):
        i = int(idx)
        if i not in picks:
            picks.append(i)
        if len(picks) >= n_applicants:
            break

    if repair_rounds:
        print(
            f"  Onarım modu AÇIK: ihlal bulunan yanıtlar en fazla {repair_rounds} "
            "kez düzeltilmek üzere ajana geri gönderilecek."
        )

    raw_audits, final_audits = [], []
    repair_stats = {"attempted": 0, "improved": 0, "fully_fixed": 0}
    for n, idx in enumerate(picks, 1):
        row = ds.X_test.iloc[idx].to_dict()
        agent.set_applicant(row, f"A-{idx:03d}")
        explanation = agent.explanation
        print(
            f"\n  [{n}/{len(picks)}] A-{idx:03d}  karar={explanation.decision} "
            f"risk={explanation.risk_probability:.1%} "
            f"({explanation.confidence_band.value})"
        )
        for question in AUDIT_QUESTIONS:
            try:
                if repair_rounds:
                    verified = agent.ask_verified(question, max_repairs=repair_rounds)
                    turn, audit, raw = verified.turn, verified.audit, verified.first_audit
                    if verified.was_repaired:
                        repair_stats["attempted"] += 1
                        if verified.improved:
                            repair_stats["improved"] += 1
                        if audit.passed:
                            repair_stats["fully_fixed"] += 1
                else:
                    turn = agent.ask_turn(question)
                    # Ajanın GERÇEKTEN çağırdığı what-if senaryolarını yeniden kur.
                    audit = audit_narrative(
                        answer=turn.answer,
                        explanation=explanation,
                        question=question,
                        used_tools=turn.tool_names,
                        what_if_results=what_ifs_from_calls(
                            agent.explainer, row, turn.tool_calls
                        ),
                    )
                    raw = audit
            except Exception as exc:  # noqa: BLE001
                print(f"      ! soru atlandı ({type(exc).__name__}: {exc})")
                continue

            raw_audits.append(raw)
            final_audits.append(audit)
            mark = "OK  " if audit.passed else "İHLAL"
            repaired = (
                f" (onarım: {raw.violations}->{audit.violations} ihlal)"
                if repair_rounds and raw.violations != audit.violations
                else ""
            )
            print(
                f"      [{mark}] tool={turn.tool_names or '—'} "
                f"temelli_sayı={audit.grounded_number_count} "
                f"ihlal={audit.violations}{repaired}"
            )
            if not audit.passed:
                if audit.ungrounded_numbers:
                    print(f"          temellenmemiş sayılar: {audit.ungrounded_numbers}")
                if audit.fabricated_concepts:
                    print(f"          uydurulmuş kavram: {audit.fabricated_concepts}")
                if audit.direction_conflicts:
                    for c in audit.direction_conflicts:
                        print(f"          yön çelişkisi: {c['feature']} "
                              f"(SHAP={c['shap_direction']}, metin={c['narrative_claim']})")
                if audit.protected_violations:
                    print(f"          korunan özellik: {audit.protected_violations}")
                if audit.missing_tool_call:
                    print("          varsayımsal soruda run_what_if ÇAĞRILMADI")

    raw_report = NarrativeFaithfulnessReport(raw_audits)
    final_report = NarrativeFaithfulnessReport(final_audits)
    d = final_report.to_dict()
    raw_d = raw_report.to_dict()

    print("\n" + "-" * 76)
    print(f"  Denetlenen yanıt        : {d['n_answers']}")
    print(
        f"  HAM sadakat skoru       : {raw_d['faithfulness_score']:.1%}  "
        f"({raw_d['n_clean']}/{raw_d['n_answers']} ihlalsiz, "
        f"{raw_d['total_violations']} ihlal)"
    )
    if repair_rounds:
        print(
            f"  ONARILMIŞ sadakat skoru : {d['faithfulness_score']:.1%}  "
            f"({d['n_clean']}/{d['n_answers']} ihlalsiz, "
            f"{d['total_violations']} ihlal)"
        )
        print(
            f"  Onarım denemesi         : {repair_stats['attempted']}, "
            f"iyileşen {repair_stats['improved']}, "
            f"tamamen düzelen {repair_stats['fully_fixed']}"
        )
    print(f"  İhlal dağılımı          : {d['violations_by_type']}")
    print(f"  KARAR                   : {d['verdict']}")

    d["raw_faithfulness"] = {
        "faithfulness_score": raw_d["faithfulness_score"],
        "n_clean": raw_d["n_clean"],
        "total_violations": raw_d["total_violations"],
        "violations_by_type": raw_d["violations_by_type"],
    }
    d["repair_enabled"] = bool(repair_rounds)
    d["repair_stats"] = repair_stats
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description="XAI-Agent sadakat + adalet denetimi")
    parser.add_argument("--skip-llm", action="store_true", help="LLM bölümünü atla")
    parser.add_argument("--n", type=int, default=5,
                        help="Ajan denetiminde kaç başvuru kullanılsın (varsayılan 5)")
    parser.add_argument("--shap-samples", type=int, default=120,
                        help="SHAP sadakat ölçümünde kaç başvuru (varsayılan 120)")
    parser.add_argument("--repair", type=int, default=0, metavar="N",
                        help=("İhlalli yanıtları en fazla N kez ajana geri "
                              "gönderip düzelttir (critic-and-revise). "
                              "0 = kapalı."))
    args = parser.parse_args()

    config.ensure_dirs()
    payload: dict = {}

    payload["shap_faithfulness"] = run_shap_section(args.shap_samples)

    if not args.skip_llm:
        narrative = run_narrative_section(args.n, repair_rounds=args.repair)
        if narrative:
            payload["narrative_faithfulness"] = narrative
    else:
        print("\n  (LLM bölümü --skip-llm ile atlandı)")

    fairness = run_fairness_audit()
    print_fairness_report(fairness)
    payload["fairness_summary"] = fairness["summary"]

    save_report(payload)
    print(f"  Sadakat raporu kaydedildi: {config.FAITHFULNESS_PATH}")
    print(f"  Adalet raporu kaydedildi : {config.FAIRNESS_PATH}\n")

    print(json.dumps(
        {
            "shap_verdict": payload["shap_faithfulness"]["verdict"],
            "narrative_score": payload.get("narrative_faithfulness", {}).get(
                "faithfulness_score"
            ),
            "narrative_score_raw": payload.get("narrative_faithfulness", {})
            .get("raw_faithfulness", {})
            .get("faithfulness_score"),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
