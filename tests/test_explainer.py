"""Faz 2 testleri: SHAP katmanı ve SHAP→JSON köprüsü.

En önemli test ``test_additivity_axiom_holds``. SHAP'ın tüm iddiası şu
eşitliğe dayanır::

    taban_deger + Σ(SHAP katkıları) = modelin ham çıktısı

Bu eşitlik tutmuyorsa açıklamalar modelin gerçek davranışını temsil etmiyor
demektir ve projenin bütün şeffaflık iddiası çöker.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_agent.schemas import ConfidenceBand, Direction, effect_strength


# --------------------------------------------------------------------------
# Çekirdek SHAP garantileri
# --------------------------------------------------------------------------
def test_additivity_axiom_holds(explainer, dataset) -> None:
    """Yerel doğruluk (local accuracy) aksiyomu tüm test setinde tutmalı."""
    result = explainer.check_additivity(dataset.X_test)
    assert result["passed"], (
        f"Toplanabilirlik ihlal edildi! maks hata = {result['max_abs_error']:.3e}"
    )
    assert result["max_abs_error"] < 1e-9
    assert result["n_samples"] == len(dataset.X_test)


def test_native_shap_matches_shap_package(explainer, dataset) -> None:
    """LightGBM'in gömülü TreeSHAP'i, ``shap`` paketiyle aynı sonucu vermeli.

    Bu projede hızlı ve kategorik-dostu olduğu için LightGBM'in kendi
    ``pred_contrib=True`` yolunu kullanıyoruz. Bu test o kararın doğruluğunu
    bağımsız bir uygulamayla çapraz doğrular.
    """
    shap = pytest.importorskip("shap")

    X = dataset.X_test.head(25)
    native, _ = explainer.contributions(X)

    tree_explainer = shap.TreeExplainer(explainer.bundle.model.booster_)
    package = np.asarray(tree_explainer.shap_values(X))

    assert package.shape == native.shape
    np.testing.assert_allclose(package, native, atol=1e-10)
    assert float(tree_explainer.expected_value) == pytest.approx(
        explainer.base_value, abs=1e-10
    )


def test_contributions_shape(explainer, dataset) -> None:
    X = dataset.X_test.head(7)
    contrib, base = explainer.contributions(X)
    assert contrib.shape == (7, len(explainer.bundle.feature_names))
    assert base.shape == (7,)
    assert len(np.unique(np.round(base, 12))) == 1, "Taban değer sabit olmalı"


# --------------------------------------------------------------------------
# Açıklama nesnesi
# --------------------------------------------------------------------------
def test_explanation_covers_all_features(sample_explanation, bundle) -> None:
    assert len(sample_explanation.all_contributions) == len(bundle.feature_names)
    assert sample_explanation.feature_names_mentioned() == set(bundle.feature_names)


def test_contributions_sorted_by_magnitude(sample_explanation) -> None:
    magnitudes = [c.abs_shap for c in sample_explanation.all_contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    ranks = [c.rank for c in sample_explanation.all_contributions]
    assert ranks == list(range(1, len(ranks) + 1))


def test_direction_matches_sign(sample_explanation) -> None:
    for c in sample_explanation.all_contributions:
        if c.shap_value > 1e-12:
            assert c.direction is Direction.INCREASES_RISK
        elif c.shap_value < -1e-12:
            assert c.direction is Direction.DECREASES_RISK
        else:
            assert c.direction is Direction.NEUTRAL


def test_shares_sum_to_hundred(sample_explanation) -> None:
    total = sum(c.share_of_total for c in sample_explanation.all_contributions)
    assert total == pytest.approx(100.0, abs=0.5)


def test_decision_matches_threshold(sample_explanation) -> None:
    expected = (
        "reddedildi"
        if sample_explanation.risk_probability >= sample_explanation.threshold
        else "onaylandı"
    )
    assert sample_explanation.decision == expected


def test_protected_features_absent_from_explanation(sample_explanation) -> None:
    assert not any(c.is_protected for c in sample_explanation.all_contributions)
    assert set(sample_explanation.model_info.excluded_protected_features) == {
        "personal_status",
        "foreign_worker",
    }


def test_explain_row_matches_explain_frame(explainer, dataset) -> None:
    frame = dataset.X_test.iloc[[5]]
    from_frame = explainer.explain_frame(frame)
    from_dict = explainer.explain_row(frame.iloc[0].to_dict())
    assert from_frame.risk_probability == pytest.approx(from_dict.risk_probability)
    assert from_frame.decision == from_dict.decision


def test_explain_frame_rejects_multirow(explainer, dataset) -> None:
    with pytest.raises(ValueError, match="tek satır bekler"):
        explainer.explain_frame(dataset.X_test.head(3))


def test_confidence_band_thresholds(explainer, dataset) -> None:
    """Eşiğe en yakın başvuru 'sınırda', en uzak olan 'net' olmalı."""
    scores = explainer.bundle.predict_proba(dataset.X_test)
    threshold = explainer.bundle.threshold
    closest = int(np.argmin(np.abs(scores - threshold)))
    farthest = int(np.argmax(np.abs(scores - threshold)))
    assert (
        explainer.explain_frame(dataset.X_test.iloc[[closest]]).confidence_band
        is ConfidenceBand.BORDERLINE
    )
    assert (
        explainer.explain_frame(dataset.X_test.iloc[[farthest]]).confidence_band
        is ConfidenceBand.CLEAR
    )


# --------------------------------------------------------------------------
# Ajan yükü (payload) — halüsinasyon önleyici tasarım
# --------------------------------------------------------------------------
def test_agent_payload_has_no_raw_logodds(sample_explanation) -> None:
    """Ham SHAP değeri ajana ULAŞMAMALI.

    Gerçek bir gözlemden doğan test: 7B'lik modele ``katki: 0.83`` verdiğimizde
    model bunu "%83 risk" diye okudu. Payload artık yalnızca yüzde biçimli
    metinler taşıyor.
    """
    payload = sample_explanation.to_agent_payload()
    for group in ("riski_artiran_etkenler", "riski_azaltan_etkenler"):
        for item in payload[group]:
            assert "katki" not in item
            assert "shap" not in " ".join(item.keys()).lower()
            for key, value in item.items():
                assert isinstance(value, str), (
                    f"'{key}' string olmalı ama {type(value)} geldi — "
                    "ajan sayı görürse yeniden yorumluyor"
                )


def test_agent_payload_percentages_are_formatted(sample_explanation) -> None:
    payload = sample_explanation.to_agent_payload()
    assert payload["risk_orani"].startswith("%")
    assert payload["karar_esigi"].startswith("%")
    for item in payload["riski_artiran_etkenler"]:
        assert item["etki_payi"].startswith("%")


def test_agent_payload_contains_rules(sample_explanation) -> None:
    """Yük, ajana verilen açık kuralları taşımalı.

    Not: Türkçe'de ``.lower()`` güvenilmez ("DEĞİLDİR".lower() birleşik nokta
    üretir), bu yüzden karşılaştırmayı ham metin üzerinde yapıyoruz.
    Metin normalleştirmesi gereken yerlerde ``faithfulness._fold`` kullanılır.
    """
    payload = sample_explanation.to_agent_payload()
    rules = " ".join(payload["KURALLAR"])
    assert "AYNEN" in rules
    assert "risk olasılığı" in rules
    assert "DEĞİLDİR" in rules


def test_effect_strength_bands() -> None:
    assert effect_strength(30) == "çok güçlü"
    assert effect_strength(18) == "güçlü"
    assert effect_strength(10) == "orta"
    assert effect_strength(2) == "zayıf"


# --------------------------------------------------------------------------
# What-if
# --------------------------------------------------------------------------
def test_what_if_actually_reruns_model(explainer, sample_applicant) -> None:
    """Vadeyi uzatmak riski artırmalı — model gerçekten yeniden koşuyor mu?"""
    result = explainer.what_if(sample_applicant, {"duration": 72})
    assert result.new_probability != result.baseline_probability
    assert result.delta_probability == pytest.approx(
        result.new_probability - result.baseline_probability
    )
    assert result.changes[0].feature == "duration"


def test_what_if_monotonic_in_duration(explainer, sample_applicant) -> None:
    """Vade uzadıkça risk (genel olarak) artmalı — mantık kontrolü."""
    short = explainer.what_if(sample_applicant, {"duration": 6}).new_probability
    long = explainer.what_if(sample_applicant, {"duration": 72}).new_probability
    assert long > short


def test_what_if_rejects_unknown_feature(explainer, sample_applicant) -> None:
    with pytest.raises(ValueError, match="Bilinmeyen"):
        explainer.what_if(sample_applicant, {"gelir": 5000})


def test_what_if_rejects_protected_feature(explainer, sample_applicant) -> None:
    """Korunan özellik modelde yok; what-if de onu değiştirmeyi reddetmeli."""
    with pytest.raises(ValueError, match="Bilinmeyen|kullanılmayan"):
        explainer.what_if(sample_applicant, {"personal_status": "male single"})


def test_what_if_requires_a_change(explainer, sample_applicant) -> None:
    with pytest.raises(ValueError, match="En az bir değişiklik"):
        explainer.what_if(sample_applicant, {})


def test_what_if_payload_is_string_only(explainer, sample_applicant) -> None:
    payload = explainer.what_if(
        sample_applicant, {"duration": 12}
    ).to_agent_payload()
    for key in ("onceki_risk_orani", "yeni_risk_orani", "karar_esigi"):
        assert payload[key].startswith("%")
    assert payload["karar_degisti_mi"] in ("EVET", "HAYIR")


# --------------------------------------------------------------------------
# Küresel önem
# --------------------------------------------------------------------------
def test_global_importance_ranks_and_sums(explainer, dataset) -> None:
    importance = explainer.global_importance(dataset.X_train)
    items = importance["features"]
    assert len(items) == len(explainer.bundle.feature_names)
    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
    assert sum(i["percent"] for i in items) == pytest.approx(100.0, abs=0.5)
    values = [i["mean_abs_shap"] for i in items]
    assert values == sorted(values, reverse=True)


def test_global_importance_not_dominated_by_one_feature(explainer, dataset) -> None:
    """Tek bir özellik etkinin yarısından fazlasını kapmamalı.

    İlk model denemesinde ``checking_status`` %50.8 pay alıyordu; bu, aşırı
    sığ (underfit) bir modelin belirtisiydi ve açıklamaları tek boyutlu
    yapıyordu. Çapraz doğrulamalı erken durdurmaya geçtikten sonra pay
    %20 civarına indi.
    """
    importance = explainer.global_importance(dataset.X_train)
    assert importance["features"][0]["percent"] < 50.0


def test_waterfall_figure_renders(explainer, sample_explanation) -> None:
    fig = explainer.waterfall_figure(sample_explanation, top_k=6)
    assert fig is not None
    assert len(fig.axes) >= 1
    import matplotlib.pyplot as plt

    plt.close(fig)
