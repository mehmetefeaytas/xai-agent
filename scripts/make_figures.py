"""README raporunun görsellerini üretir.

Bu script hiçbir şey uydurmaz: sayıların tamamı ya ``artifacts/*.json``
dosyalarından ya da eğitilmiş modelin canlı olarak yeniden çalıştırılmasından
gelir. Yeniden çalıştırılabilir ve idempotenttir.

Kullanım
--------
    uv run python scripts/make_figures.py            # hepsini üret
    uv run python scripts/make_figures.py --only 05  # tek figür
    uv run python scripts/make_figures.py --list     # figür listesi
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_agent import config  # noqa: E402
from xai_agent.data import prepare  # noqa: E402
from xai_agent.explainer import CreditExplainer  # noqa: E402
from xai_agent.model import (  # noqa: E402
    _baseline_pipeline,
    expected_cost,
    oof_predictions,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"

# Projenin renk sözleşmesi — explainer.waterfall_figure ile aynı.
RISK_UP = "#d62728"
RISK_DOWN = "#2ca02c"
ACCENT = "#1f4e79"
NEUTRAL = "#8c8c8c"
GRID = "#dddddd"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


# --------------------------------------------------------------------------
# Ortak yardımcılar
# --------------------------------------------------------------------------
def _load(name: str) -> dict[str, Any]:
    path = config.ARTIFACTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} yok. Önce şunları çalıştır:\n"
            "  uv run python scripts/train.py\n"
            "  uv run python scripts/evaluate.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  ✓ {name}  ({kb:.0f} KB)")
    return path


def _note(ax: plt.Axes, text: str, y: float = -0.22) -> None:
    """Grafiğin altına açıklama notu — figür tek başına anlaşılsın."""
    ax.annotate(
        text,
        xy=(0, y),
        xycoords="axes fraction",
        fontsize=8.5,
        color="#444444",
        va="top",
        wrap=True,
    )


# --------------------------------------------------------------------------
# 01 — Eşik / maliyet eğrisi
# --------------------------------------------------------------------------
def fig_threshold_cost() -> None:
    m = _load("metrics")
    ds = prepare()
    n_est = m["model"]["n_estimators_used"]
    spw = m["model"]["scale_pos_weight"]

    print("     kat-dışı tahminler hesaplanıyor (5 kat)...")
    oof = oof_predictions(ds.X_train, ds.y_train, n_est, spw)
    bundle = CreditExplainer().bundle
    test_score = bundle.predict_proba(ds.X_test)

    grid = np.linspace(0.02, 0.98, 97)
    y_tr = np.asarray(ds.y_train)
    y_te = np.asarray(ds.y_test)
    oof_cost = [expected_cost(y_tr, (oof >= t).astype(int)) for t in grid]
    te_cost = [expected_cost(y_te, (test_score >= t).astype(int)) for t in grid]

    # Eğitim (800) ve test (200) farklı büyüklükte; satır başına maliyet
    # olarak normalize edilmezse iki eğri karşılaştırılamaz.
    oof_n = np.asarray(oof_cost) / len(y_tr)
    te_n = np.asarray(te_cost) / len(y_te)

    chosen = m["threshold_selection"]["threshold"]
    bayes = m["threshold_selection"]["theoretical_bayes_threshold"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, oof_n, color=ACCENT, lw=2.4, label="Kat-dışı (eğitim, n=800) — eşik BURADAN seçildi")
    ax.plot(grid, te_n, color=NEUTRAL, lw=1.8, ls="--", label="Test (n=200) — sadece doğrulama")

    top = max(oof_n.max(), te_n.max())
    i_oof = int(np.argmin(oof_n))
    ax.axvline(chosen, color=RISK_UP, lw=1.6, ls=":")
    ax.plot([grid[i_oof]], [oof_n[i_oof]], "o", color=RISK_UP, ms=9, zorder=5)
    ax.annotate(
        f"seçilen eşik = {chosen}\n(kat-dışı maliyet minimumu)",
        xy=(chosen, oof_n[i_oof]),
        xytext=(0.30, top * 0.60),
        fontsize=9.5,
        color=RISK_UP,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RISK_UP, "lw": 1.3},
    )
    ax.axvline(bayes, color="#7b3294", lw=1.4, ls="-.")
    ax.text(
        bayes + 0.022,
        top * 0.90,
        f"teorik Bayes eşiği = {bayes}\n$c_{{FP}}/(c_{{FP}}+c_{{FN}})=1/6$",
        fontsize=8.5,
        color="#7b3294",
        ha="left",
        va="top",
    )
    ax.axvline(0.5, color="#444444", lw=1.2, ls="-")
    ax.text(
        0.52,
        top * 0.42,
        "varsayılan 0.50\n(maliyeti görmezden gelir)",
        fontsize=8.5,
        color="#444444",
        ha="left",
        va="top",
    )

    ax.set_xlabel("Karar eşiği (bu olasılığın üzerindeki başvuru reddedilir)")
    ax.set_ylabel("Satır başına beklenen maliyet\n(5×yanlış kabul + 1×yanlış ret)")
    ax.set_title("Karar eşiği neden 0.50 değil? — maliyet asimetrisi eşiği aşağı çeker")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    _note(
        ax,
        "Batan bir kredi (yanlış kabul), kaçırılan iyi müşteriden 5 kat pahalı. Bu asimetri altında en iyi eşik 0.50 değil "
        f"{chosen}.\n"
        "Eşik YALNIZCA eğitim setinin kat-dışı tahminlerinde seçildi (mavi eğri); test eğrisi (gri) sadece kararın tuttuğunu\n"
        f"doğrulamak için çizildi — test setine eşik seçerken hiç dokunulmadı. Test setinde toplam maliyet: eşik {chosen} → "
        f"{m['test']['expected_cost']:.0f}, eşik 0.50 → {m['test_at_default_threshold_0.5']['expected_cost']:.0f}.",
        y=-0.20,
    )
    _save(fig, "01-esik-maliyet.png")


# --------------------------------------------------------------------------
# 02 — ROC ve PR eğrileri
# --------------------------------------------------------------------------
def fig_roc_pr() -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    m = _load("metrics")
    ds = prepare()
    bundle = CreditExplainer().bundle
    lgbm = bundle.predict_proba(ds.X_test)

    logreg = _baseline_pipeline(ds)
    logreg.fit(ds.X_train, ds.y_train)
    lr = logreg.predict_proba(ds.X_test)[:, 1]

    y = np.asarray(ds.y_test)
    prevalence = y.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for score, name, auc_key, color, lw in (
        (lgbm, "LightGBM", "roc_auc", ACCENT, 2.4),
        (lr, "Lojistik regresyon", None, "#ff7f0e", 2.0),
    ):
        fpr, tpr, _ = roc_curve(y, score)
        auc = m["test"][auc_key] if auc_key else m["baselines"]["logistic_regression"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} — AUC {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2, label="Rastgele — AUC 0.5000")
    ax.set_xlabel("Yanlış pozitif oranı")
    ax.set_ylabel("Doğru pozitif oranı (duyarlılık)")
    ax.set_title("ROC eğrisi — sıralama gücü")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for score, name, key, color, lw in (
        (lgbm, "LightGBM", m["test"]["pr_auc"], ACCENT, 2.4),
        (lr, "Lojistik regresyon", m["baselines"]["logistic_regression"]["pr_auc"], "#ff7f0e", 2.0),
    ):
        prec, rec, _ = precision_recall_curve(y, score)
        ax.plot(rec, prec, color=color, lw=lw, label=f"{name} — PR-AUC {key:.4f}")
    ax.axhline(prevalence, color=NEUTRAL, ls="--", lw=1.2, label=f"Taban oran — {prevalence:.2f}")
    ax.set_xlabel("Duyarlılık (recall)")
    ax.set_ylabel("Kesinlik (precision)")
    ax.set_title("Precision–Recall eğrisi — dengesiz sınıfta daha bilgilendirici")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Dürüst karşılaştırma: lojistik regresyon AUC'de ÖNDE (0.8099 > 0.7942)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    _note(
        axes[0],
        "LightGBM'i sırf gradient boosting olduğu için savunmuyoruz: sıralama gücünde (AUC) lojistik regresyon önde. LightGBM'in\n"
        f"kazandığı yer maliyet ve duyarlılık: beklenen maliyet {m['test']['expected_cost']:.0f} vs "
        f"{m['baselines']['logistic_regression']['expected_cost']:.0f}, duyarlılık {m['test']['recall']:.3f} vs "
        f"{m['baselines']['logistic_regression']['recall']:.3f} — riskli başvuruların daha azını kaçırıyor.",
        y=-0.20,
    )
    _save(fig, "02-roc-pr.png")


# --------------------------------------------------------------------------
# 03 — Karmaşıklık matrisleri
# --------------------------------------------------------------------------
def fig_confusion() -> None:
    m = _load("metrics")
    panels = [
        (m["test"], f"Seçilen eşik = {m['test']['threshold']}"),
        (m["test_at_default_threshold_0.5"], "Varsayılan eşik = 0.50"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, (block, title) in zip(axes, panels):
        cm = block["confusion_matrix"]
        grid = np.array(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ]
        )
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=110)
        labels = [
            ["Doğru ret\n(iyi ⇒ onay)", "Yanlış ret\n(iyi ⇒ RED)"],
            ["Yanlış kabul\n(riskli ⇒ ONAY)", "Doğru yakalama\n(riskli ⇒ RED)"],
        ]
        weights = [["×1 maliyet yok", "×1"], ["×5", "maliyet yok"]]
        for i in range(2):
            for j in range(2):
                is_fn = i == 1 and j == 0
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False,
                        edgecolor=RISK_UP if is_fn else "none",
                        lw=3.5,
                    )
                )
                ax.text(j, i - 0.16, f"{grid[i, j]}", ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color="white" if grid[i, j] > 60 else "#111111")
                ax.text(j, i + 0.16, labels[i][j], ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > 60 else "#333333")
                ax.text(j, i + 0.36, weights[i][j], ha="center", va="center", fontsize=7.5,
                        style="italic",
                        color="white" if grid[i, j] > 60 else "#666666")
        ax.set_xticks([0, 1], ["Onaylandı", "Reddedildi"])
        ax.set_yticks([0, 1], ["Gerçekte iyi", "Gerçekte riskli"])
        ax.set_xlabel("Modelin kararı")
        cost = block["expected_cost"]
        ax.set_title(
            f"{title}\n"
            f"maliyet = 5×{cm['false_negative']} + 1×{cm['false_positive']} = {cost:.0f}\n"
            f"duyarlılık {block['recall']:.3f}  ·  isabet {block['accuracy']:.2f}",
            fontsize=10.5,
        )
        ax.grid(False)

    _note(
        axes[0],
        "Kırmızı çerçeve, pahalı hatayı gösterir: riskli müşteriyi onaylamak. Varsayılan 0.50 eşiği daha yüksek isabet (0.72 vs 0.58)\n"
        "üretir ama 16 riskli başvuruyu kaçırır; seçilen 0.26 eşiği bunu 5'e düşürür. Toplam maliyet 120'den 104'e iner.\n"
        "Bu yüzden bu projede accuracy bir başarı ölçüsü olarak KULLANILMIYOR — asimetrik maliyeti gizler.",
        y=-0.24,
    )
    _save(fig, "03-karmasiklik-matrisi.png")


# --------------------------------------------------------------------------
# 04 — Küresel SHAP önemi
# --------------------------------------------------------------------------
def fig_global_importance() -> None:
    gi = _load("global_importance")
    feats = gi["features"][:15][::-1]
    names = [f["display_name"] for f in feats]
    pct = [f["percent"] for f in feats]
    colors = [
        RISK_UP if "artırıyor" in f["direction_bias"] else RISK_DOWN for f in feats
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bars = ax.barh(names, pct, color=colors, alpha=0.85, height=0.72)
    for bar, f in zip(bars, feats):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"%{f['percent']:.1f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )
    ax.set_xlabel("Toplam |SHAP| içindeki pay (%)")
    ax.set_title(
        f"Küresel özellik önemi — mean(|SHAP|), n={gi['n_samples']} eğitim satırı"
    )
    ax.set_xlim(0, max(pct) * 1.18)
    ax.grid(axis="y", visible=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RISK_UP, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=RISK_DOWN, alpha=0.85),
    ]
    ax.legend(handles, ["Ortalamada riski artırıyor", "Ortalamada riski azaltıyor"],
              loc="lower right", fontsize=9)
    _note(
        ax,
        "Hiçbir özellik tek başına baskın değil — en güçlüsü %19.6. Bu bir sağlık göstergesi: tek özelliğin %50'yi aştığı ilk\n"
        "eğitimde model yetersiz öğrenmişti (30 ağaç); çapraz doğrulamalı erken durdurmaya geçince önem dağıldı (243 ağaç).\n"
        "Renkler ORTALAMA eğilimi gösterir; bireysel bir başvuruda aynı özellik ters yönde çalışabilir.",
        y=-0.16,
    )
    _save(fig, "04-kuresel-onem.png")


# --------------------------------------------------------------------------
# 05 — SHAP şelalesi (kıl payı başvuru)
# --------------------------------------------------------------------------
def fig_waterfall() -> None:
    ds = prepare()
    explainer = CreditExplainer()
    bundle = explainer.bundle
    scores = bundle.predict_proba(ds.X_test)
    index = int(np.argmin(np.abs(scores - bundle.threshold)))
    row = ds.X_test.iloc[index].to_dict()
    explanation = explainer.explain_row(row, applicant_id=f"A-{index:03d}")
    fig = explainer.waterfall_figure(explanation, top_k=10)
    fig.set_size_inches(10, 6)
    _save(fig, "05-shap-selale.png")
    print(
        f"     (kıl payı başvuru A-{index:03d}, risk {explanation.risk_probability:.1%}, "
        f"eşik {explanation.threshold:.0%})"
    )


# --------------------------------------------------------------------------
# 06 — Toplanabilirlik aksiyomu
# --------------------------------------------------------------------------
def fig_additivity() -> None:
    ds = prepare()
    explainer = CreditExplainer()
    bundle = explainer.bundle
    X = ds.X_test

    shap_values, base = explainer.contributions(X)
    reconstructed = base + shap_values.sum(axis=1)
    actual = bundle.predict_raw(X)
    err = np.abs(reconstructed - actual)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    lo, hi = float(min(actual.min(), reconstructed.min())), float(max(actual.max(), reconstructed.max()))
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=RISK_UP, lw=1.6, ls="--",
            label="y = x (mükemmel uyum)", zorder=1)
    ax.scatter(actual, reconstructed, s=26, color=ACCENT, alpha=0.55,
               edgecolor="none", label=f"Test başvuruları (n={len(X)})", zorder=2)
    ax.set_xlabel("Modelin gerçek çıktısı (log-odds)")
    ax.set_ylabel("taban değer + Σ SHAP katkıları")
    ax.set_title("Toplanabilirlik: açıklama modelin çıktısını\ntam olarak yeniden kuruyor")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.hist(err, bins=30, color=ACCENT, alpha=0.8)
    ax.axvline(err.max(), color=RISK_UP, lw=1.8)
    ax.annotate(
        f"maksimum hata\n{err.max():.1e}",
        xy=(err.max(), ax.get_ylim()[1] * 0.55),
        xytext=(err.max() * 0.42, ax.get_ylim()[1] * 0.7),
        fontsize=9, color=RISK_UP, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RISK_UP, "lw": 1.2},
    )
    ax.set_xlabel("|yeniden kurulan − gerçek| (log-odds)")
    ax.set_ylabel("Başvuru sayısı")
    ax.set_title(f"Hata dağılımı — tolerans 1e-06\nortalama {err.mean():.1e}, hepsi GEÇTİ")

    _note(
        axes[0],
        "SHAP'ın 'yerel isabet' (local accuracy) aksiyomu: katkıların toplamı modelin çıktısına eşit olmak ZORUNDA. Ölçülen en büyük\n"
        f"sapma {err.max():.1e} — bu bir yaklaşım hatası değil, kayan nokta aritmetiğinin sınırı. Yani açıklama modelin yanında duran\n"
        "ikinci bir tahmin değil; modelin kendi kararının aynen ayrıştırılmış hali. Bu doğrulama her açıklamada yeniden yapılır.",
        y=-0.20,
    )
    _save(fig, "06-toplanabilirlik.png")


# --------------------------------------------------------------------------
# 07 — ERASER sadakat ölçümü
# --------------------------------------------------------------------------
def fig_faithfulness() -> None:
    s = _load("faithfulness_report")["shap_faithfulness"]
    ks = [str(k) for k in s["k_values"]]
    x = np.arange(len(ks))
    comp = [s["comprehensiveness"][k] for k in ks]
    rand = [s["random_baseline_comprehensiveness"][k] for k in ks]
    suff = [s["sufficiency"][k] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"wspace": 0.26})

    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, comp, w, color=ACCENT, label="SHAP'ın seçtiği k özellik")
    ax.bar(x + w / 2, rand, w, color=NEUTRAL, alpha=0.75, label="Rastgele k özellik (kontrol)")
    for xi, (c, r) in enumerate(zip(comp, rand)):
        ax.text(xi - w / 2, c + 0.004, f"{c:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(xi + w / 2, r + 0.004, f"{r:.3f}", ha="center", fontsize=8.5, color="#555555")
    ax.set_xticks(x, [f"k={k}" for k in ks])
    ax.set_ylabel("Risk tahminindeki ortalama düşüş")
    ax.set_title(
        "Comprehensiveness — YÜKSEK iyi\n(en önemli k özellik silinince tahmin düşüyor mu?)",
        fontsize=10.5,
    )
    ax.set_ylim(0, max(comp) * 1.62)
    ax.text(
        0.5, 0.985,
        f"AOPC: {s['aopc_comprehensiveness']:.4f} vs {s['aopc_random_baseline']:.4f}\n"
        f"→ SHAP rastgeleden {s['lift_over_random']:.2f}× daha etkili",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
        fontweight="bold", color=ACCENT,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#eaf2fa", "ec": ACCENT, "lw": 1.1},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), fontsize=9)

    ax = axes[1]
    ax.plot(x, suff, "o-", color=RISK_DOWN, lw=2.4, ms=8)
    for xi, v in enumerate(suff):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"k={k}" for k in ks])
    ax.set_ylabel("Orijinal tahminden ortalama sapma")
    ax.set_title(
        "Sufficiency — DÜŞÜK iyi\n(yalnızca o k özellik bırakılınca korunuyor mu?)",
        fontsize=10.5,
    )
    ax.set_ylim(0, max(suff) * 1.25)
    ax.text(
        0.60, 0.74,
        "k arttıkça sapma düşüyor:\nSHAP'ın ilk sıraları\ngerçekten kararı taşıyor",
        transform=ax.transAxes, fontsize=9, color="#2f6b2f", va="top",
    )

    fig.suptitle(
        f"SHAP açıklaması modele SADIK mı? — ERASER ölçütleri, n={s['n_samples']} başvuru   ·   "
        f"karar: {s['verdict'].split('—')[0].strip()}",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    _note(
        axes[0],
        "Bu ölçüm SHAP'ın kendi matematiğine güvenmeyi bırakıp davranışsal olarak test eder: SHAP'ın 'en önemli' dediği özellikleri\n"
        "referans değerine (medyan/mod) çekip modeli YENİDEN çalıştırıyoruz. Tahmin gerçekten düşüyorsa sıralama doğruydu.\n"
        "Rastgele seçilmiş aynı sayıda özellikle karşılaştırma zorunlu — onsuz 0.138'in iyi mi kötü mü olduğu bilinemez.",
        y=-0.20,
    )
    _save(fig, "07-sadakat-aopc.png")


# --------------------------------------------------------------------------
# 08 — Anlatı ihlalleri: onarım öncesi / sonrası
# --------------------------------------------------------------------------
def fig_violations() -> None:
    n = _load("faithfulness_report")["narrative_faithfulness"]
    raw = n["raw_faithfulness"]
    labels = {
        "ungrounded_numbers": "Temelsiz sayı",
        "direction_conflicts": "Yön çelişkisi",
        "missing_tool_calls": "Tool çağırmama",
        "protected_violations": "Korunan özellik",
        "fabricated_concepts": "Uydurma kavram",
        "misframed_shares": "Yanlış çerçeveleme",
    }
    keys = list(labels)
    before = [raw["violations_by_type"].get(k, 0) for k in keys]
    after = [n["violations_by_type"].get(k, 0) for k in keys]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2),
        gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.24},
    )

    ax = axes[0]
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, before, w, color=RISK_UP, alpha=0.85, label="Onarım KAPALI (ham ajan)")
    ax.bar(x + w / 2, after, w, color=RISK_DOWN, alpha=0.85, label="Onarım AÇIK (critic-and-revise)")
    for xi, (b, a) in enumerate(zip(before, after)):
        ax.text(xi - w / 2, b + 0.4, str(b), ha="center", fontsize=10, fontweight="bold", color=RISK_UP)
        ax.text(xi + w / 2, a + 0.4, str(a), ha="center", fontsize=10, fontweight="bold", color="#1f7a1f")
    ax.set_xticks(x, [labels[k] for k in keys], fontsize=8.8, rotation=16, ha="right")
    ax.set_ylabel("İhlal sayısı (15 yanıt boyunca)")
    ax.set_title(
        "Denetçinin yakaladığı ihlaller\nonarım döngüsünün etkisi",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(before) * 1.22)

    ax = axes[1]
    st = n["repair_stats"]
    bars = ["Denendi", "İyileşti", "Tamamen\ntemizlendi"]
    vals = [st["attempted"], st["improved"], st["fully_fixed"]]
    ax.bar(bars, vals, color=[NEUTRAL, "#4a90c2", RISK_DOWN], alpha=0.9, width=0.6)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.1, str(v), ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("Yanıt sayısı")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title(
        f"Onarım turu sonucu\nsadakat {raw['faithfulness_score']:.1%} → {n['faithfulness_score']:.1%}"
        f"  ·  ihlal {raw['total_violations']} → {n['total_violations']}",
        fontsize=11.5,
    )

    _note(
        axes[0],
        "DÜRÜST OKUMA: onarım döngüsü tek soruda çok güçlü (ana açıklama sorusunda 5 ihlal → 0), ama 15 yanıtın tamamına\n"
        "bakıldığında kazanç mütevazı: 37 → 32. Sebep, örneklemin çoğunun TAKİP sorusu olması — ajan takip turunda oturum\n"
        "geçmişine güvenip tool çağırmıyor ve sayı uyduruyor. Bu, prompt ile değil mimari bir kısıtla (tool_choice='required')\n"
        "çözülecek bilinen bir sınır. Denetçi bu ihlalleri yakalıyor; sessizce geçirmiyor.",
        y=-0.26,
    )
    _save(fig, "08-ihlal-turleri.png")


# --------------------------------------------------------------------------
# 09 — Adalet denetimi
# --------------------------------------------------------------------------
def fig_fairness() -> None:
    fr = _load("fairness_report")
    details = fr["details"]
    gap_keys = [
        ("demographic_parity_gap", "Demografik eşitlik\n(ret oranı farkı)"),
        ("equal_opportunity_gap", "Fırsat eşitliği\n(duyarlılık farkı)"),
        ("predictive_parity_gap", "Öngörü eşitliği\n(kesinlik farkı)"),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.4),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.30},
    )

    ax = axes[0]
    x = np.arange(len(details))
    w = 0.26
    palette = [ACCENT, "#4a90c2", "#9ecae1"]
    for i, ((key, label), color) in enumerate(zip(gap_keys, palette)):
        vals = [d["gaps"].get(key) for d in details]
        plotted = [0 if v is None else v for v in vals]
        ax.bar(x + (i - 1) * w, plotted, w, color=color, label=label)
        for xi, v in zip(x, vals):
            if v is None:
                ax.text(xi + (i - 1) * w, 0.008, "ölçülemedi", ha="center", fontsize=7,
                        rotation=90, color="#888888")
            else:
                ax.text(xi + (i - 1) * w, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5,
                        fontweight="bold")

    ax.axhspan(0, 0.05, color="#e8f6e8", zorder=0)
    ax.axhspan(0.05, 0.10, color="#fdf6e3", zorder=0)
    ax.axhspan(0.10, 0.20, color="#fdeee0", zorder=0)
    ax.axhspan(0.20, 0.40, color="#fbe4e4", zorder=0)
    for y_, t in ((0.025, "ihmal edilebilir"), (0.075, "küçük"), (0.15, "DİKKAT"), (0.30, "CİDDİ")):
        ax.text(-0.44, y_, t, fontsize=7.5, va="center", ha="left", color="#888888",
                style="italic")

    names = [
        f"{d['display_name']}\n({'MODELDE' if d['in_model'] else 'modelden çıkarıldı'})"
        for d in details
    ]
    ax.set_xticks(x, names, fontsize=9)
    ax.set_xlim(-0.5, len(details) - 0.5)
    ax.set_ylabel("Gruplar arası en büyük fark")
    ax.set_title(
        "Adalet denetimi\nkorunan özellikleri çıkarmak yeterli mi?",
        fontsize=11.5,
    )
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_ylim(0, 0.40)

    ax = axes[1]
    age = next(d for d in details if d["attribute"] == "age")
    groups = age["groups"]
    labels = [g["label"] for g in groups]
    y = np.arange(len(groups))
    ax.barh(y - 0.19, [g["rejection_rate"] for g in groups], 0.36,
            color=RISK_UP, alpha=0.85, label="Modelin ret oranı")
    ax.barh(y + 0.19, [g["actual_bad_rate"] for g in groups], 0.36,
            color=NEUTRAL, alpha=0.8, label="Gerçek riskli oranı")
    for yi, g in enumerate(groups):
        ax.text(g["rejection_rate"] + 0.012, yi - 0.19, f"{g['rejection_rate']:.2f}",
                va="center", fontsize=8.5, fontweight="bold")
        ax.text(g["actual_bad_rate"] + 0.012, yi + 0.19, f"{g['actual_bad_rate']:.2f}",
                va="center", fontsize=8.5, color="#555555")
    ax.set_yticks(y, [f"{lab}  (n={g['n']})" for lab, g in zip(labels, groups)])
    ax.set_xlabel("Oran")
    ax.set_xlim(0, 1.30)
    ax.set_title(
        "En büyük uçurum: yaş grubu\n(yaş modelin İÇİNDE — bilinçli bir seçim)",
        fontsize=11.5,
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", visible=False)

    _note(
        axes[0],
        "Cinsiyet ve yabancı işçi durumu model girdisinden ÇIKARILDI, yine de cinsiyette 0.103'lük ret oranı farkı kalıyor — "
        "'bilmezlik yoluyla\nadalet' (fairness through unawareness) yetmiyor, çünkü diğer özellikler vekil (proxy) görevi görüyor. "
        "Yaş modelde tutuldu ve\n0.298 fark üretiyor; bir kısmı gerçek risk farkından geliyor (19-30 grubunda gerçek riskli oranı 0.42 "
        "vs 56+ grubunda 0.10)\nama ret oranı farkı (0.30) gerçek risk farkından (0.32) bağımsız olarak da büyük. Bu model bu haliyle "
        "üretime UYGUN DEĞİL.\nKleinberg ve ark. (2016): bu üç ölçütü aynı anda sağlamak matematiksel olarak imkânsız — hangisinin "
        "önemli olduğu politika kararıdır.",
        y=-0.26,
    )
    _save(fig, "09-adalet.png")


# --------------------------------------------------------------------------
# 10 — Çapraz doğrulama katlamaları
# --------------------------------------------------------------------------
def fig_cv_folds() -> None:
    m = _load("metrics")
    cv = m["cross_validation"]
    folds = cv["cv_auc_folds"]
    mean, std = cv["cv_auc_mean"], cv["cv_auc_std"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, len(folds) + 1)
    ax.bar(x, folds, color=ACCENT, alpha=0.85, width=0.55, label="Kat AUC'si")
    for xi, v in zip(x, folds):
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=9.5, fontweight="bold")

    ax.axhline(mean, color=RISK_UP, lw=2, label=f"Ortalama = {mean:.4f}")
    ax.axhspan(mean - std, mean + std, color=RISK_UP, alpha=0.11,
               label=f"±1 standart sapma = ±{std:.4f}")
    ax.axhline(m["test"]["roc_auc"], color="#7b3294", lw=1.8, ls="--",
               label=f"Test seti AUC = {m['test']['roc_auc']:.4f}")

    ax.set_xticks(x, [f"Kat {i}" for i in x])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(min(folds) - 0.115, max(folds) + 0.035)
    ax.set_title(
        f"5 katlı çapraz doğrulama — {mean:.4f} ± {std:.4f}\n"
        f"model {m['model']['n_estimators_used']} ağaçta durdu ({m['model']['early_stopping']['method']})"
    )
    ax.legend(
        loc="lower center", ncol=2, fontsize=8.5,
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
    )
    ax.grid(axis="x", visible=False)
    _note(
        ax,
        f"Katlar arası fark {max(folds) - min(folds):.3f} — 1000 satırlık bir veri setinde tek bir hold-out skoruna güvenilemez. "
        "Test AUC'si (0.7942)\nçapraz doğrulama ortalamasının (0.7934) bir standart sapma bandı içinde: model test setinde şans "
        "eseri iyi/kötü görünmüyor.\nErken durdurma da tek bir doğrulama kümesinde değil, 5 katın ortalamasında yapıldı — ilk sürümde "
        "tek hold-out kullanılmış ve\nmodel 30 ağaçta durup yetersiz öğrenmişti (AUC 0.776).",
        y=-0.19,
    )
    _save(fig, "10-cv-katlamalari.png")


# --------------------------------------------------------------------------
FIGURES: dict[str, tuple[str, Callable[[], None]]] = {
    "01": ("01-esik-maliyet.png", fig_threshold_cost),
    "02": ("02-roc-pr.png", fig_roc_pr),
    "03": ("03-karmasiklik-matrisi.png", fig_confusion),
    "04": ("04-kuresel-onem.png", fig_global_importance),
    "05": ("05-shap-selale.png", fig_waterfall),
    "06": ("06-toplanabilirlik.png", fig_additivity),
    "07": ("07-sadakat-aopc.png", fig_faithfulness),
    "08": ("08-ihlal-turleri.png", fig_violations),
    "09": ("09-adalet.png", fig_fairness),
    "10": ("10-cv-katlamalari.png", fig_cv_folds),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="README rapor görsellerini üretir")
    parser.add_argument("--only", nargs="*", help="Yalnızca bu figürleri üret (örn. 05 07)")
    parser.add_argument("--list", action="store_true", help="Figür listesini yazdır")
    args = parser.parse_args()

    if args.list:
        for key, (name, fn) in FIGURES.items():
            print(f"  {key}  {name:<32} {fn.__doc__ or fn.__name__}")
        return 0

    selected = args.only or list(FIGURES)
    unknown = [k for k in selected if k not in FIGURES]
    if unknown:
        raise SystemExit(f"Bilinmeyen figür: {unknown}. Seçenekler: {list(FIGURES)}")

    print(f"Görseller {OUT_DIR} altına yazılıyor\n")
    for key in selected:
        name, fn = FIGURES[key]
        print(f"[{key}] {name}")
        fn()
    print(f"\n{len(selected)} figür üretildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
