"""Model katmanı: LightGBM eğitimi, baseline karşılaştırması, metrikler.

Neden LightGBM?
---------------
Tabular (tablo biçimli) kredi verisinde gradient boosting hâlâ sinir
ağlarını yenen yaklaşım. LightGBM'i özellikle seçtik çünkü **kategorik
özellikleri doğrudan** işleyebiliyor: ``purpose`` gibi 10 seviyeli bir
sütunu one-hot encoding ile 10 ayrı sütuna açmamız gerekmiyor.

Bu, XAI açısından kritik bir avantaj. One-hot yapsaydık SHAP bize
``purpose_used_car = +0.1``, ``purpose_new_car = -0.05`` gibi 10 parça
verecekti ve bunları tekrar birleştirmemiz gerekecekti. Şimdi tek bir
``purpose`` özelliği için tek bir SHAP değeri alıyoruz — açıklaması da
doğrudan insan diline çevrilebiliyor.

Neden accuracy'ye bakmıyoruz?
-----------------------------
Veri %70 "good" / %30 "bad". Her şeye "good" diyen aptal bir model %70
accuracy alır. Bu yüzden ana metriklerimiz **ROC-AUC** (sıralama kalitesi)
ve **PR-AUC** (azınlık sınıfını yakalama kalitesi). Ayrıca UCI'nin resmî
maliyet matrisiyle (kötüyü iyi sanmak 5 kat pahalı) **maliyet-optimal
karar eşiği** öğreniyoruz — 0.5 eşiği bu problemde yanlıştır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import Dataset, align_frame, prepare

_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 4)


# --------------------------------------------------------------------------
# Paket: model + meta veri birlikte taşınır
# --------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Eğitilmiş model ve onu doğru kullanmak için gereken her şey.

    Modeli tek başına kaydetmek yetmez: hangi sütun sırasıyla eğitildiği,
    hangi karar eşiğinin kullanılacağı ve metrikleri de yanında taşınmalı.
    Aksi hâlde çıkarım anında sessiz hatalar oluşur.
    """

    model: lgb.LGBMClassifier
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    threshold: float
    metrics: dict[str, Any] = field(default_factory=dict)
    n_estimators_used: int = 0

    # -- tahmin ------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Riskli olma (``bad``) olasılığını döndürür."""
        X = self._align(X)
        return self.model.predict_proba(X)[:, 1]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Ham log-odds skorunu döndürür (SHAP toplanabilirlik kontrolü için)."""
        X = self._align(X)
        return np.asarray(self.model.booster_.predict(X, raw_score=True), dtype=float)

    def predict_label(self, X: pd.DataFrame) -> np.ndarray:
        """Karar eşiğine göre 0/1 etiketi döndürür."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def decide(self, X: pd.DataFrame) -> list[str]:
        """İnsan-okunur karar: ``'reddedildi'`` veya ``'onaylandı'``."""
        return [
            "reddedildi" if p >= self.threshold else "onaylandı"
            for p in self.predict_proba(X)
        ]

    def predict_one(self, row: dict[str, object]) -> dict[str, Any]:
        """Tek bir başvuru sözlüğü için tam karar paketi."""
        frame = align_frame(row, self.feature_names)
        proba = float(self.predict_proba(frame)[0])
        raw = float(self.predict_raw(frame)[0])
        return {
            "risk_probability": proba,
            "raw_logodds": raw,
            "threshold": self.threshold,
            "decision": "reddedildi" if proba >= self.threshold else "onaylandı",
            "margin": proba - self.threshold,
        }

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sütun sırasını modelin eğitildiği sıraya zorlar."""
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Girdi çerçevesinde eksik sütun(lar): {missing}")
        return X[list(self.feature_names)]

    # -- kalıcılık ---------------------------------------------------------
    def save(self, path=None) -> None:
        import joblib

        path = path or config.MODEL_PATH
        config.ensure_dirs()
        joblib.dump(self, path)

    @staticmethod
    def load(path=None) -> ModelBundle:
        import joblib

        path = path or config.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {path}\n"
                "Önce eğitimi çalıştırın:  uv run python scripts/train.py"
            )
        return joblib.load(path)


# --------------------------------------------------------------------------
# Metrikler
# --------------------------------------------------------------------------
def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """UCI maliyet matrisine göre toplam hata maliyeti.

    Kötü müşteriyi iyi sanmak (FN) 5 birim, iyi müşteriyi kötü sanmak (FP)
    1 birim maliyetli. Bu asimetri kredi riskinde gerçektir: batan kredi,
    kaçırılan iyi müşteriden çok daha pahalıdır.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(config.COST_FALSE_NEGATIVE * fn + config.COST_FALSE_POSITIVE * fp)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov–Smirnov istatistiği — kredi skorlamada standart ayrım ölçüsü."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def find_cost_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """Beklenen maliyeti en küçük yapan karar eşiğini arar.

    Returns:
        ``(eşik, o eşikteki maliyet)``
    """
    best_t, best_c = 0.5, float("inf")
    for t in _THRESHOLD_GRID:
        cost = expected_cost(y_true, (y_score >= t).astype(int))
        if cost < best_c:
            best_t, best_c = float(t), cost
    return best_t, best_c


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Bir tahmin kümesi için tüm metrikleri hesaplar."""
    y_true = np.asarray(y_true)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "expected_cost": expected_cost(y_true, y_pred),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# Baseline modeller
# --------------------------------------------------------------------------
def _baseline_pipeline(ds: Dataset) -> Pipeline:
    """Lojistik regresyon baseline'ı: kategorikler için one-hot gerekir."""
    cat = [c for c in ds.feature_names if c in ds.categorical_features]
    num = [c for c in ds.feature_names if c not in ds.categorical_features]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def train_baselines(ds: Dataset) -> dict[str, dict[str, Any]]:
    """İki baseline eğitir: her şeye çoğunluk diyen aptal model ve lojistik regresyon.

    Baseline olmadan "AUC 0.78 iyi mi?" sorusunun cevabı yok. Aptal model
    0.50 verir; lojistik regresyon ~0.75 verir. LightGBM'in kattığı değeri
    ancak bu iki referansla ölçebiliriz.
    """
    results: dict[str, dict[str, Any]] = {}

    dummy = DummyClassifier(strategy="prior", random_state=config.RANDOM_SEED)
    dummy.fit(ds.X_train, ds.y_train)
    dummy_score = dummy.predict_proba(ds.X_test)[:, 1]
    results["dummy_prior"] = compute_metrics(ds.y_test, dummy_score, 0.5)

    logreg = _baseline_pipeline(ds)
    # Eşiği kat-dışı tahminlerle seç: LightGBM'e uyguladığımız aynı kural.
    lr_oof = cross_val_predict(
        logreg,
        ds.X_train,
        ds.y_train,
        cv=StratifiedKFold(
            config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
        ),
        method="predict_proba",
    )[:, 1]
    lr_t, _ = find_cost_optimal_threshold(np.asarray(ds.y_train), lr_oof)
    logreg.fit(ds.X_train, ds.y_train)
    lr_score = logreg.predict_proba(ds.X_test)[:, 1]
    results["logistic_regression"] = compute_metrics(ds.y_test, lr_score, lr_t)

    return results


# --------------------------------------------------------------------------
# Eğitim
# --------------------------------------------------------------------------
def positive_class_weight(y: pd.Series | np.ndarray) -> float:
    """``scale_pos_weight`` = negatif/pozitif oranı.

    Veri %70 good / %30 bad olduğu için pozitif (riskli) sınıfa ~2.33 kat
    ağırlık verilir. Bu olmadan model azınlık sınıfını görmezden gelmeye
    meyleder.
    """
    y = np.asarray(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return round(n_neg / max(n_pos, 1.0), 6)


def _make_estimator(
    n_estimators: int | None = None, scale_pos_weight: float | None = None
) -> lgb.LGBMClassifier:
    params = dict(config.LGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return lgb.LGBMClassifier(**params)


def find_best_n_estimators_cv(
    X: pd.DataFrame, y: pd.Series, scale_pos_weight: float
) -> dict[str, Any]:
    """Ağaç sayısını **çapraz doğrulamalı** erken durdurma ile belirler.

    Neden tek doğrulama seti değil?
    -------------------------------
    800 satırlık eğitim setinden %20 ayırınca elde 160 satır kalıyor. Bu kadar
    küçük bir sette AUC eğrisi gürültülüdür; erken durdurma tesadüfi bir
    dalgalanmayı "iyileşme durdu" sanıp modeli 30 ağaçta kesebiliyor (ilk
    denemede tam bu oldu ve model underfit kaldı).

    ``lgb.cv`` her turda 5 katın **ortalamasına** bakar. Gürültü 5 kat
    ortalamasında büyük ölçüde sönümlenir, dolayısıyla durdurma noktası çok
    daha kararlı olur.
    """
    native = {
        k: v
        for k, v in config.LGBM_PARAMS.items()
        if k not in ("n_estimators", "random_state", "n_jobs", "verbosity")
    }
    native.update(
        {
            "scale_pos_weight": scale_pos_weight,
            "seed": config.RANDOM_SEED,
            "num_threads": 0,
            "verbose": -1,
            "metric": "auc",
        }
    )

    dtrain = lgb.Dataset(X, label=np.asarray(y), free_raw_data=False)
    hist = lgb.cv(
        native,
        dtrain,
        num_boost_round=config.LGBM_PARAMS["n_estimators"],
        nfold=config.CV_FOLDS,
        stratified=True,
        shuffle=True,
        seed=config.RANDOM_SEED,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    key = next(k for k in hist if k.endswith("auc-mean"))
    curve = list(hist[key])
    best_iter = int(np.argmax(curve)) + 1
    return {
        "best_n_estimators": max(best_iter, 30),
        "cv_auc_at_best": round(float(curve[best_iter - 1]), 4),
        "rounds_evaluated": len(curve),
        "method": f"lgb.cv, {config.CV_FOLDS} kat, erken durdurma 80 tur",
    }


def oof_predictions(
    X: pd.DataFrame, y: pd.Series, n_estimators: int, scale_pos_weight: float
) -> np.ndarray:
    """Eğitim seti için kat-dışı (out-of-fold) olasılık tahminleri.

    Karar eşiğini bu tahminler üzerinden seçiyoruz. Böylece hem test setine
    hiç dokunmuyoruz (sızıntı yok) hem de eşiği 160 satır değil **800**
    satırın tamamı üzerinden belirliyoruz.
    """
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    oof = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        est = _make_estimator(n_estimators, scale_pos_weight)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = est.predict_proba(X.iloc[va_idx])[:, 1]
    return oof


def cross_val_auc(
    ds: Dataset, n_estimators: int, scale_pos_weight: float
) -> dict[str, float]:
    """Eğitim seti üzerinde 5-kat çapraz doğrulama AUC'si.

    Tek bir test skoru şansa açıktır (200 satır!). CV, modelin kararlılığını
    gösterir: standart sapma büyükse skora güvenmemek gerekir.
    """
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    aucs: list[float] = []
    for tr_idx, va_idx in skf.split(ds.X_train, ds.y_train):
        est = _make_estimator(n_estimators, scale_pos_weight)
        est.fit(ds.X_train.iloc[tr_idx], ds.y_train.iloc[tr_idx])
        proba = est.predict_proba(ds.X_train.iloc[va_idx])[:, 1]
        aucs.append(float(roc_auc_score(ds.y_train.iloc[va_idx], proba)))
    return {
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
        "cv_auc_folds": [round(a, 4) for a in aucs],
    }


def train(ds: Dataset | None = None, verbose: bool = True) -> ModelBundle:
    """Tüm eğitim akışı: erken durdurma → yeniden eğitim → eşik → metrikler.

    Akış:
      1. Eğitim setini içeriden fit/valid olarak böl.
      2. Valid üzerinde AUC ile erken durdurma yaparak ideal ağaç sayısını bul.
      3. O ağaç sayısıyla **tüm** eğitim setinde yeniden eğit (veriyi harcamamak için).
      4. Valid üzerinde maliyet-optimal karar eşiğini belirle (test'e dokunmadan!).
      5. Test setinde nihai metrikleri hesapla.
    """
    ds = ds or prepare()
    spw = positive_class_weight(ds.y_train)

    # 1) Ağaç sayısı: çapraz doğrulamalı erken durdurma
    es = find_best_n_estimators_cv(ds.X_train, ds.y_train, spw)
    best_iter = es["best_n_estimators"]

    # 2) Karar eşiği: kat-dışı tahminler üzerinden maliyet minimizasyonu
    oof = oof_predictions(ds.X_train, ds.y_train, best_iter, spw)
    threshold, oof_cost = find_cost_optimal_threshold(np.asarray(ds.y_train), oof)

    # 3) Nihai model: tüm eğitim setinde yeniden eğit
    final = _make_estimator(best_iter, spw)
    final.fit(ds.X_train, ds.y_train)

    bundle = ModelBundle(
        model=final,
        feature_names=ds.feature_names,
        categorical_features=ds.categorical_features,
        threshold=threshold,
        n_estimators_used=best_iter,
    )

    # 5) Metrikler
    test_score = bundle.predict_proba(ds.X_test)
    train_score = bundle.predict_proba(ds.X_train)

    metrics: dict[str, Any] = {
        "dataset": ds.summary(),
        "model": {
            "family": "LightGBM (LGBMClassifier)",
            "n_estimators_used": best_iter,
            "scale_pos_weight": spw,
            "params": {
                k: v for k, v in config.LGBM_PARAMS.items() if k != "n_estimators"
            },
            "early_stopping": es,
            "cost_matrix": {
                "false_negative": config.COST_FALSE_NEGATIVE,
                "false_positive": config.COST_FALSE_POSITIVE,
            },
        },
        "threshold_selection": {
            "method": "out-of-fold cost minimisation (5 kat)",
            "threshold": round(threshold, 4),
            "oof_cost": oof_cost,
            "oof_roc_auc": round(float(roc_auc_score(ds.y_train, oof)), 4),
            "theoretical_bayes_threshold": round(
                config.COST_FALSE_POSITIVE
                / (config.COST_FALSE_POSITIVE + config.COST_FALSE_NEGATIVE),
                4,
            ),
            "note": (
                "Eşik yalnızca eğitim setinin kat-dışı tahminlerinde seçildi; "
                "test seti hiç kullanılmadı (veri sızıntısı yok)."
            ),
        },
        "train": compute_metrics(ds.y_train, train_score, threshold),
        "test": compute_metrics(ds.y_test, test_score, threshold),
        "cross_validation": cross_val_auc(ds, best_iter, spw),
        "baselines": train_baselines(ds),
    }
    metrics["test_at_default_threshold_0.5"] = compute_metrics(
        ds.y_test, test_score, 0.5
    )
    bundle.metrics = metrics

    if verbose:
        _print_report(metrics)

    return bundle


def _print_report(m: dict[str, Any]) -> None:
    t, b = m["test"], m["baselines"]
    print("\n" + "=" * 68)
    print("  FAZ 1 — LightGBM Kredi Risk Modeli / Test Seti Sonuçları")
    print("=" * 68)
    print(f"  Veri            : {m['dataset']['n_train']} eğitim / "
          f"{m['dataset']['n_test']} test, {m['dataset']['n_features']} özellik")
    print(f"  Dışlanan (etik) : {m['dataset']['dropped_protected']}")
    print(f"  Ağaç sayısı     : {m['model']['n_estimators_used']} (erken durdurma)")
    print(f"  Karar eşiği     : {t['threshold']}  (maliyet-optimal, 0.5 DEĞİL)")
    print("-" * 68)
    print(f"  {'Metrik':<22}{'LightGBM':>12}{'LogReg':>12}{'Dummy':>12}")
    print("-" * 68)
    for key, label in [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("ks", "KS"),
        ("recall", "Recall (bad)"),
        ("precision", "Precision (bad)"),
        ("f1", "F1 (bad)"),
        ("accuracy", "Accuracy"),
        ("brier", "Brier (düşük iyi)"),
        ("expected_cost", "Maliyet (düşük iyi)"),
    ]:
        print(
            f"  {label:<22}{t[key]:>12}"
            f"{b['logistic_regression'][key]:>12}"
            f"{b['dummy_prior'][key]:>12}"
        )
    cv = m["cross_validation"]
    print("-" * 68)
    print(f"  5-kat CV AUC    : {cv['cv_auc_mean']} ± {cv['cv_auc_std']}")
    cm = t["confusion_matrix"]
    print(f"  Karmaşıklık mat.: TN={cm['true_negative']} FP={cm['false_positive']} "
          f"FN={cm['false_negative']} TP={cm['true_positive']}")
    print("=" * 68 + "\n")


def save_metrics(bundle: ModelBundle) -> None:
    """Metrikleri ``artifacts/metrics.json`` dosyasına yazar."""
    config.ensure_dirs()
    config.METRICS_PATH.write_text(
        json.dumps(bundle.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_bundle(path=None) -> ModelBundle:
    """Kaydedilmiş modeli yükler (kısayol)."""
    return ModelBundle.load(path)
