"""Faz 1 testleri: veri katmanı ve model.

Buradaki testlerin çoğu "sessiz hata" avcısı. Örneğin kategori sırasının
kayması ya da sütun sırasının değişmesi hiçbir istisna atmaz — model sadece
yanlış tahmin eder. Bu tür hataları ancak açık testlerle yakalayabiliriz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_agent import config
from xai_agent.data import align_frame, encode_target, load_raw, prepare
from xai_agent.features import (
    ALL_FEATURES,
    CATEGORICAL_LEVELS,
)
from xai_agent.model import (
    expected_cost,
    find_cost_optimal_threshold,
    ks_statistic,
    positive_class_weight,
)


# --------------------------------------------------------------------------
# Veri
# --------------------------------------------------------------------------
def test_raw_data_shape() -> None:
    df = load_raw()
    assert len(df) == 1000, "German Credit 1000 satır olmalı"
    assert config.TARGET_COL in df.columns
    for feature in ALL_FEATURES:
        assert feature in df.columns, f"'{feature}' ham veride yok"


def test_categorical_levels_are_pinned() -> None:
    """Kategori sırası features.py'deki sırayla BİREBİR aynı olmalı.

    Bu test kritik: LightGBM kategorileri pandas'ın ``category`` kodlarına
    göre öğrenir. Sıra kayarsa model sessizce yanlış tahmin yapar.
    """
    df = load_raw()
    for col, levels in CATEGORICAL_LEVELS.items():
        assert isinstance(df[col].dtype, pd.CategoricalDtype), (
            f"'{col}' kategorik tipte değil"
        )
        assert tuple(df[col].cat.categories) == levels, (
            f"'{col}' kategori sırası bozuk: "
            f"{tuple(df[col].cat.categories)} != {levels}"
        )


def test_unknown_category_raises() -> None:
    """Sözlükte olmayan bir kategori değeri sessizce NaN'a düşmemeli."""
    from xai_agent.data import _coerce_dtypes

    df = load_raw().copy()
    df["housing"] = df["housing"].astype(str)
    df.loc[0, "housing"] = "kesinlikle-olmayan-deger"
    with pytest.raises(ValueError, match="sözlükte tanımlı olmayan"):
        _coerce_dtypes(df)


def test_target_encoding_direction() -> None:
    """``1 = bad`` sözleşmesi — SHAP işaretlerinin yönü buna bağlı."""
    y = encode_target(pd.Series(["good", "bad", "good"]))
    assert list(y) == [0, 1, 0]


def test_protected_features_excluded_by_default(dataset) -> None:
    for feature in config.PROTECTED_FEATURES:
        assert feature not in dataset.feature_names, (
            f"Korunan özellik '{feature}' modelde olmamalı"
        )
        assert feature in dataset.protected_train.columns, (
            f"'{feature}' adalet denetimi için saklanmalı"
        )


def test_split_is_stratified(dataset) -> None:
    """Eğitim ve test setlerinin sınıf oranları birbirine yakın olmalı."""
    assert abs(dataset.y_train.mean() - dataset.y_test.mean()) < 0.02


def test_split_sizes(dataset) -> None:
    assert dataset.n_train == 800
    assert dataset.n_test == 200
    assert len(dataset.feature_names) == len(ALL_FEATURES) - len(
        config.PROTECTED_FEATURES
    )


def test_split_is_deterministic() -> None:
    """Aynı tohumla iki kez bölünce aynı sonuç gelmeli (yeniden üretilebilirlik)."""
    a, b = prepare(), prepare()
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


# --------------------------------------------------------------------------
# align_frame — what-if ve manuel girişin kalbi
# --------------------------------------------------------------------------
def test_align_frame_orders_columns(sample_applicant, bundle) -> None:
    shuffled = dict(reversed(list(sample_applicant.items())))
    frame = align_frame(shuffled, bundle.feature_names)
    assert tuple(frame.columns) == bundle.feature_names


def test_align_frame_rejects_invalid_category(sample_applicant, bundle) -> None:
    bad = dict(sample_applicant)
    bad["housing"] = "sarayda"
    with pytest.raises(ValueError, match="geçersiz değer"):
        align_frame(bad, bundle.feature_names)


def test_align_frame_rejects_missing_feature(sample_applicant, bundle) -> None:
    incomplete = dict(sample_applicant)
    incomplete.pop("age")
    with pytest.raises(ValueError, match="Eksik özellik"):
        align_frame(incomplete, bundle.feature_names)


# --------------------------------------------------------------------------
# Metrik yardımcıları
# --------------------------------------------------------------------------
def test_expected_cost_uses_asymmetric_matrix() -> None:
    """Kötüyü iyi sanmak (FN), iyiyi kötü sanmaktan (FP) 5 kat pahalı olmalı."""
    y_true = np.array([1, 0])
    only_fn = expected_cost(y_true, np.array([0, 0]))  # 1 FN
    only_fp = expected_cost(y_true, np.array([1, 1]))  # 1 FP
    assert only_fn == config.COST_FALSE_NEGATIVE
    assert only_fp == config.COST_FALSE_POSITIVE
    assert only_fn == 5 * only_fp


def test_perfect_prediction_costs_nothing() -> None:
    y = np.array([0, 1, 1, 0])
    assert expected_cost(y, y) == 0.0


def test_cost_optimal_threshold_is_low() -> None:
    """5:1 maliyet oranında optimal eşik teorik olarak 1/(1+5)≈0.167 civarıdır.

    Bulunan eşiğin 0.5'in belirgin biçimde altında olması, maliyet
    mantığının gerçekten çalıştığının kanıtı.
    """
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=600)
    score = np.clip(y * 0.4 + rng.normal(0.3, 0.2, size=600), 0, 1)
    threshold, cost = find_cost_optimal_threshold(y, score)
    assert 0.05 <= threshold < 0.5
    assert cost >= 0


def test_ks_statistic_bounds() -> None:
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.5, size=200)
    assert 0.0 <= ks_statistic(y, rng.random(200)) <= 1.0
    assert ks_statistic(y, y.astype(float)) == pytest.approx(1.0)


def test_positive_class_weight() -> None:
    y = np.array([0] * 70 + [1] * 30)
    assert positive_class_weight(y) == pytest.approx(70 / 30, rel=1e-6)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def test_bundle_metadata(bundle) -> None:
    assert bundle.n_estimators_used > 0
    assert 0.0 < bundle.threshold < 1.0
    assert len(bundle.feature_names) == 18
    assert bundle.metrics, "Metrikler paketle birlikte kaydedilmiş olmalı"


def test_threshold_is_not_naive_half(bundle) -> None:
    """Maliyet matrisi 5:1 iken 0.5 eşiği yanlıştır; eğitim bunu bulmalı."""
    assert bundle.threshold < 0.5


def test_predictions_are_valid_probabilities(bundle, dataset) -> None:
    proba = bundle.predict_proba(dataset.X_test)
    assert proba.shape == (len(dataset.X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predictions_are_deterministic(bundle, dataset) -> None:
    first = bundle.predict_proba(dataset.X_test.head(20))
    second = bundle.predict_proba(dataset.X_test.head(20))
    np.testing.assert_array_equal(first, second)


def test_column_order_does_not_change_prediction(bundle, dataset) -> None:
    """Sütunlar karışık gelse bile ``_align`` doğru sırayı dayatmalı."""
    X = dataset.X_test.head(10)
    shuffled = X[list(reversed(list(X.columns)))]
    np.testing.assert_allclose(
        bundle.predict_proba(X), bundle.predict_proba(shuffled)
    )


def test_missing_column_raises(bundle, dataset) -> None:
    with pytest.raises(ValueError, match="eksik sütun"):
        bundle.predict_proba(dataset.X_test.drop(columns=["age"]))


def test_model_beats_dummy_baseline(bundle) -> None:
    """LightGBM, "her şeye çoğunluk de" modelini yenmeli."""
    test = bundle.metrics["test"]
    dummy = bundle.metrics["baselines"]["dummy_prior"]
    assert test["roc_auc"] > 0.70, f"AUC beklenenden düşük: {test['roc_auc']}"
    assert test["roc_auc"] > dummy["roc_auc"] + 0.15
    assert test["expected_cost"] < dummy["expected_cost"]


def test_model_beats_logreg_on_cost(bundle) -> None:
    """Asıl önemli metrikte (maliyet) LightGBM lojistik regresyonu yenmeli.

    Not: German Credit gibi 1000 satırlık tablo verisinde lojistik regresyon
    AUC'de LightGBM'e yakın hatta üstün çıkabilir. Bu bilinen bir durumdur ve
    ``artifacts/metrics.json`` içinde dürüstçe raporlanır. Karar kalitesini
    belirleyen maliyet metriğinde ise LightGBM öne geçiyor.
    """
    test = bundle.metrics["test"]
    logreg = bundle.metrics["baselines"]["logistic_regression"]
    assert test["expected_cost"] <= logreg["expected_cost"]


def test_cross_validation_is_stable(bundle) -> None:
    cv = bundle.metrics["cross_validation"]
    assert cv["cv_auc_mean"] > 0.70
    assert cv["cv_auc_std"] < 0.10, "CV varyansı çok yüksek — model kararsız"


def test_threshold_near_theoretical_bayes(bundle) -> None:
    """Ampirik eşik, teorik Bayes eşiğinin makul yakınında olmalı."""
    selection = bundle.metrics["threshold_selection"]
    theoretical = selection["theoretical_bayes_threshold"]
    assert theoretical == pytest.approx(1 / 6, abs=0.01)
    assert abs(selection["threshold"] - theoretical) < 0.25


def test_predict_one_returns_consistent_decision(bundle, sample_applicant) -> None:
    result = bundle.predict_one(sample_applicant)
    expected = "reddedildi" if result["risk_probability"] >= bundle.threshold else "onaylandı"
    assert result["decision"] == expected
    assert result["margin"] == pytest.approx(
        result["risk_probability"] - bundle.threshold
    )
