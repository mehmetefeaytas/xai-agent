"""Faz 5 testleri: sadakat denetçisinin kendisi doğru çalışıyor mu?

Bu dosya "denetçiyi denetler". Bir sadakat denetçisi her şeye "geçti" diyorsa
işe yaramaz. Bu yüzden her ihlal türü için **kasıtlı olarak bozuk** bir
anlatı üretip denetçinin onu yakaladığını doğruluyoruz.
"""

from __future__ import annotations

import pytest

from xai_agent.faithfulness import (
    _split_sentences,
    allowed_numbers,
    audit_narrative,
    evaluate_shap_faithfulness,
    reference_row,
)


# --------------------------------------------------------------------------
# Cümle bölme — sessiz bir hatanın regresyon testi
# --------------------------------------------------------------------------
def test_sentence_split_preserves_decimals() -> None:
    """Ondalık sayılar cümle sınırı sanılmamalı.

    Bu bir regresyon testi. İlk uygulamada ``re.split(r"[.!?\\n]+")``
    kullanıyorduk ve "%21.6" ifadesi "riski %21" + "6 oranında artırıyor"
    diye ikiye bölünüyordu. Sonuç: cümle düzeyindeki tüm denetimler sessizce
    hiçbir şey kontrol etmiyordu.
    """
    sentences = _split_sentences("Risk oranı %44.4. Eşik %26.0 idi. Bitti.")
    assert len(sentences) == 3
    assert "44.4" in sentences[0]
    assert "26.0" in sentences[1]


def test_sentence_split_handles_comma_decimals() -> None:
    sentences = _split_sentences("Değer 21,6 oldu. Sonra bitti.")
    assert len(sentences) == 2
    assert "21.6" in sentences[0]


def test_fold_handles_turkish_dotted_capital_i() -> None:
    """Türkçe 'İ' harfinin ``.lower()`` tuzağına karşı regresyon testi.

    ``"DEĞİLDİR".lower()`` Python'da 'i' + U+0307 (birleşik nokta) üretir,
    bu yüzden düz karşılaştırma başarısız olur. ``_fold`` bunu düzeltmeli.
    """
    from xai_agent.faithfulness import _fold

    assert "değildir".lower() not in "DEĞİLDİR".lower()  # tuzak gerçek
    assert _fold("DEĞİLDİR") == _fold("değildir")  # _fold onu kapatıyor
    assert _fold("Riski ARTIRIYOR") == _fold("riski artırıyor")


# --------------------------------------------------------------------------
# İzin verilen sayılar kümesi
# --------------------------------------------------------------------------
def test_allowed_numbers_includes_payload_values(sample_explanation) -> None:
    allowed = allowed_numbers(sample_explanation)
    assert round(sample_explanation.risk_probability * 100, 1) in allowed
    assert round(sample_explanation.threshold * 100, 1) in allowed
    for c in sample_explanation.all_contributions:
        assert round(c.share_of_total, 1) in allowed


def test_allowed_numbers_includes_what_if(explainer, sample_applicant,
                                          sample_explanation) -> None:
    result = explainer.what_if(sample_applicant, {"duration": 30})
    allowed = allowed_numbers(sample_explanation, [result])
    assert round(result.new_probability * 100, 1) in allowed


# --------------------------------------------------------------------------
# İhlal tespiti — her tür için kasıtlı bozuk anlatı
# --------------------------------------------------------------------------
def _audit(text: str, explanation, question="test", tools=("get_decision_explanation",)):
    return audit_narrative(text, explanation, question=question, used_tools=list(tools))


def test_clean_narrative_passes(sample_explanation) -> None:
    e = sample_explanation
    c = e.all_contributions[0]
    text = (
        f"Başvuru {e.decision}. Model risk oranını %{e.risk_probability * 100:.1f} "
        f"olarak hesapladı; karar eşiği %{e.threshold * 100:.1f}. "
        f"{c.display_name} ({c.value_label}), kararı belirleyen etkenlerin "
        f"%{c.share_of_total:.1f}'ini oluşturuyor."
    )
    audit = _audit(text, e)
    assert audit.passed, audit.to_dict()
    assert audit.grounded_number_count >= 3


def test_detects_ungrounded_number(sample_explanation) -> None:
    audit = _audit("Risk oranınız %99.7 olarak hesaplandı.", sample_explanation)
    assert not audit.passed
    assert 99.7 in audit.ungrounded_numbers


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Geliriniz yetersiz olduğu için reddedildi.", "gelir"),
        ("Kredi notunuz çok düşük.", "kredi notu"),
        ("Findeks kaydınız olumsuz.", "findeks"),
        ("İcra kaydınız bulunuyor.", "icra"),
    ],
)
def test_detects_fabricated_concepts(sample_explanation, text, expected) -> None:
    audit = _audit(text, sample_explanation)
    assert not audit.passed
    assert expected in audit.fabricated_concepts


def test_detects_direction_conflict(sample_explanation) -> None:
    """SHAP 'azaltıyor' derken metin 'artırıyor' diyorsa yakalanmalı."""
    from xai_agent.schemas import Direction

    protective = next(
        c for c in sample_explanation.all_contributions
        if c.direction is Direction.DECREASES_RISK
    )
    audit = _audit(
        f"{protective.display_name} riski artırıyor.", sample_explanation
    )
    assert not audit.passed
    assert audit.direction_conflicts
    assert audit.direction_conflicts[0]["feature"] == protective.feature


def _opposite_claim(contribution) -> str:
    from xai_agent.schemas import Direction

    return (
        "artırıyor"
        if contribution.direction is Direction.DECREASES_RISK
        else "azaltıyor"
    )


def test_direction_check_tolerates_inflected_names(sample_explanation) -> None:
    """Ekler ve büyük/küçük harf değişimi yön denetimini bozmamalı.

    Ajan "Mevcut işte çalışma süresi" yerine "mevcut ÇALIŞMA süreniz" yazabilir.
    Ayırt edici kelime ("çalışma") korunduğu sürece eşleşme tutmalı.
    """
    target = next(
        (c for c in sample_explanation.all_contributions if c.feature == "employment"),
        None,
    )
    if target is None:
        pytest.skip("employment bu açıklamada yok")
    audit = _audit(
        f"Mevcut ÇALIŞMA süreniz riski {_opposite_claim(target)}.",
        sample_explanation,
    )
    assert audit.direction_conflicts, "çekimli ad yakalanamadı"
    assert audit.direction_conflicts[0]["feature"] == "employment"


def test_direction_check_does_not_confuse_similar_names(sample_explanation) -> None:
    """Yalnızca genel kelimeleri paylaşan özellikler birbirine karışmamalı.

    Bu, ölçülmüş bir yanlış pozitifin regresyon testi. İlk uygulamada eşleşme
    "adın kelimelerinin çoğunluğu" kuralına dayanıyordu ve
    "Mevcut **iş**te çalışma **süresi**" ile "Mevcut adre**ste** ikamet
    **süresi**" birbirine karışıyordu; 5 başvuruluk bir denetimde 25 uydurma
    yön çelişkisi üretti. Artık yalnızca *ayırt edici* kelimeler sayılıyor
    (bkz. ``_GENERIC_NAME_WORDS``).
    """
    from xai_agent.faithfulness import _fold, best_feature_match

    sentence = _fold(
        "Mevcut adreste ikamet süresi (kademe): 4 kademe olma durumunuz, "
        "riski azaltıyor."
    )
    match = best_feature_match(sentence, sample_explanation.all_contributions)
    assert match is not None, "cümledeki özellik hiç bulunamadı"
    assert match.feature == "residence_since", (
        f"yanlış özellikle eşleşti: {match.feature}"
    )


def test_conditional_sentences_are_not_direction_claims(sample_explanation) -> None:
    """Tavsiye/varsayım cümleleri mevcut SHAP yönü hakkında iddia değildir.

    "Vadeyi düşürerek riski azaltabilirsiniz" cümlesi, vadenin şu anda riski
    azalttığını söylemiyor. İlk uygulamada bu tür cümleler yön çelişkisi
    sayılıyordu.
    """
    target = next(
        (c for c in sample_explanation.all_contributions if c.feature == "duration"),
        None,
    )
    if target is None:
        pytest.skip("duration bu açıklamada yok")
    audit = _audit(
        "Kredi vadesini düşürerek riski azaltabilirsiniz.", sample_explanation
    )
    assert not audit.direction_conflicts


def test_temporal_conditional_is_not_a_direction_claim(sample_explanation) -> None:
    """"-dığında / -duğunuzda" eki senaryo anlatır, mevcut yönü iddia etmez.

    Ölçülmüş yanlış pozitif: "Kredi vadesini 12 ay olarak düşürdüğünüzde risk
    oranı %88.2'den %70.4'e düştü" cümlesi, vadenin şu anda riski azalttığını
    söylemiyor — bir what-if sonucunu aktarıyor. İlk sürüm bunu 8 kez yön
    çelişkisi saydı.
    """
    audit = _audit(
        "Kredi vadesini 12 ay olarak düşürdüğünüzde risk oranı düştü.",
        sample_explanation,
        question="Vade 12 aya düşse ne olur?",
        tools=("run_what_if",),
    )
    assert not audit.direction_conflicts


def test_numbers_from_the_question_are_grounded(sample_explanation) -> None:
    """Kullanıcının sorduğu sayıyı tekrar etmek halüsinasyon değildir."""
    audit = _audit(
        "Vadenin 36 aya çıkması senaryosunu çalıştırdım.",
        sample_explanation,
        question="Vade 36 aya çıksa ne olur?",
        tools=("run_what_if",),
    )
    assert 36.0 not in audit.ungrounded_numbers


def test_feature_vocabulary_is_masked_before_fabrication_check(
    sample_explanation,
) -> None:
    """Meşru bir özellik adı yasaklı kelime içerse alarm çalmamalı.

    ``installment_commitment`` özelliğinin Türkçe adı "Taksit yükü (gelire
    oran kademesi)" ve içinde "gelir" geçiyor. Ajan bu özelliği DOĞRU biçimde
    andığında denetçi "gelir uydurdu" diyordu — ölçülmüş bir yanlış pozitif.
    """
    audit = _audit(
        "Taksit yükü (gelire oran kademesi) kararın bir bölümünü oluşturuyor.",
        sample_explanation,
    )
    assert "gelir" not in audit.fabricated_concepts

    # Buna karşın gerçek uydurma hâlâ yakalanmalı
    still_caught = _audit("Geliriniz yetersiz görüldü.", sample_explanation)
    assert "gelir" in still_caught.fabricated_concepts


def test_verb_idiom_gelir_is_not_a_fabricated_concept(sample_explanation) -> None:
    """«…anlamına gelir» fiilini gelir (income) kavramı sanmamalı.

    Türkçe'de "gelir" hem bir isim (income) hem "gelmek" fiilinin geniş zaman
    çekimidir. Alt-dizi araması ikisini ayırt edemiyordu.

    Bu yanlış pozitif, tanıtım videosunun çekimi sırasında gerçek bir ajan
    yanıtında yakalandı; denetçi tek ihlal olarak bunu raporlamıştı.
    """
    audit = _audit(
        "Risk oranı eşiğin altında, bu da riski daha düşük bir seviyeye "
        "indirgeyen bir karar anlamına gelir.",
        sample_explanation,
    )
    assert "gelir" not in audit.fabricated_concepts
    assert audit.fabricated_contexts == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Bu sonuç, başvurunun onaylandığı anlamına gelir.",
        "Eşiğin altında kalmak olumlu bir sonuç manasına gelir.",
        "Bu durum ortaya gelir gelmez karar yeniden hesaplanır.",
    ],
)
def test_verb_idioms_do_not_trigger_fabrication(sample_explanation, sentence) -> None:
    """Ölçülen deyim ailesinin tamamı alarm üretmemeli."""
    assert "gelir" not in _audit(sentence, sample_explanation).fabricated_concepts


def test_income_fabrication_still_caught_next_to_verb_idiom(
    sample_explanation,
) -> None:
    """Deyim maskeleme, GERÇEK uydurmayı gizlemeye dönüşmemeli.

    Aynı yanıtta hem deyim hem gerçek uydurma varsa ihlal yine raporlanmalı;
    aksi hâlde düzeltme kolaylıkla bir kaçış yoluna dönüşür.
    """
    audit = _audit(
        "Bu, kararın onaylandığı anlamına gelir. Ayrıca aylık geliriniz "
        "yetersiz bulundu.",
        sample_explanation,
    )
    assert "gelir" in audit.fabricated_concepts


def test_detects_misframed_share(sample_explanation) -> None:
    """Etki payını risk artış miktarı gibi sunmak ihlaldir.

    Gerçek gözlem: ajan "%13.7 oranında riski artırıyor" yazdı. Sayı
    temellidir (bir etki payıdır) ama ANLAMI çarpıtılmıştır — gerçek risk
    oranı %44.4'tü. Sayı denetimi bunu yakalamaz.
    """
    c = sample_explanation.all_contributions[0]
    audit = _audit(
        f"{c.display_name} riski %{c.share_of_total:.1f} oranında artırıyor.",
        sample_explanation,
    )
    assert audit.misframed_shares, "çerçeveleme hatası yakalanamadı"


def test_detects_misframed_share_number_first(sample_explanation) -> None:
    """"%21.3 riski artırıyor" — sayı, risk kelimesinden ÖNCE gelen kalıp.

    Ölçümden doğan regresyon testi: qwen2.5 en sık bu sırayı kuruyor ve ilk
    regex yalnızca "riski %21.3 ... artırıyor" sırasını yakalıyordu.
    """
    c = sample_explanation.all_contributions[0]
    audit = _audit(
        f"Bu etken %{c.share_of_total:.1f} riski artırıyor.", sample_explanation
    )
    assert audit.misframed_shares, "sayı-önce kalıbı yakalanamadı"


def test_applicant_id_digits_are_grounded(explainer, dataset) -> None:
    """Başvuru kimliğindeki sayı "temellenmemiş" sayılmamalı.

    Ölçülmüş yanlış pozitif: ajan "Başvuru kimliği A-046 için karar..." yazdığında
    denetçi 46 sayısını uydurma sayıyordu. Kimlik yükün içinde yer alıyor.
    """
    explanation = explainer.explain_frame(
        dataset.X_test.iloc[[46]], applicant_id="A-046"
    )
    audit = _audit("Başvuru kimliği A-046 için karar verildi.", explanation)
    assert 46.0 not in audit.ungrounded_numbers
    assert audit.passed, audit.to_dict()


def test_correct_share_framing_passes(sample_explanation) -> None:
    c = sample_explanation.all_contributions[0]
    audit = _audit(
        f"{c.display_name}, kararın %{c.share_of_total:.1f}'ini oluşturuyor.",
        sample_explanation,
    )
    assert not audit.misframed_shares


def test_real_risk_rate_framing_passes(sample_explanation) -> None:
    """Gerçek risk oranını fiille birleştirmek meşrudur."""
    e = sample_explanation
    audit = _audit(
        f"Bu başvuruda model riski %{e.risk_probability * 100:.1f} olarak "
        "hesapladı ve bu eşiği aşıyor.",
        e,
    )
    assert not audit.misframed_shares


def test_detects_language_drift(sample_explanation) -> None:
    """İngilizce'ye kayan yanıt ihlal sayılmalı.

    Ölçülmüş davranış: onarım turundan sonra 7B model İngilizce'ye kayıp tool
    çıktısını "Feature: ... / Impact Share: ..." biçiminde döktü. Sayılar
    doğruydu ama ürün Türkçe konuşan bir başvuru sahibine hitap ediyor;
    İngilizce bir yanıt teknik olarak sadık olsa bile kullanılamaz.
    """
    audit = _audit(
        "Based on the provided tool response: Decision: Approved. "
        "Feature: Vadesiz hesap durumu, Value: bakiye, Impact Share: 16.4%",
        sample_explanation,
    )
    assert audit.language_drift


def test_turkish_answer_has_no_language_drift(sample_explanation) -> None:
    e = sample_explanation
    c = e.all_contributions[0]
    audit = _audit(
        f"Başvuru {e.decision}. Model risk oranını "
        f"%{e.risk_probability * 100:.1f} olarak hesapladı. "
        f"{c.display_name}, kararın %{c.share_of_total:.1f}'ini oluşturuyor.",
        e,
    )
    assert not audit.language_drift
    assert audit.passed, audit.to_dict()


def test_repair_prompt_asks_for_turkish_and_prose() -> None:
    """Onarım mesajı dil ve biçim talimatını tekrar etmeli.

    Sistem promptu Türkçe istiyor ama 7B model son mesaja ağırlık veriyor;
    düzeltme turunda hatırlatılmazsa İngilizce'ye ve alan-listesi biçimine
    kayıyor. Ölçümde gözlendi.
    """
    from xai_agent.prompts import build_repair_prompt

    class Audit:
        used_tools = ["get_decision_explanation"]
        ungrounded_numbers = [99.7]
        fabricated_concepts: list = []
        direction_conflicts: list = []
        misframed_shares: list = []
        protected_violations: list = []
        language_drift = False
        missing_tool_call = False

    message = build_repair_prompt(Audit())
    assert "TÜRKÇE yaz" in message
    assert "alan-alan listeleme" in message


def test_detects_protected_attribute_reasoning(sample_explanation) -> None:
    audit = _audit(
        "Kadın olmanız nedeniyle riskiniz yüksek değerlendirildi.",
        sample_explanation,
    )
    assert not audit.passed
    assert audit.protected_violations


def test_excluding_protected_attribute_is_allowed(sample_explanation) -> None:
    """Korunan özelliği 'kullanmıyoruz' demek DOĞRU davranıştır."""
    audit = _audit(
        "Bu model cinsiyet ve uyruk bilgisini hiç kullanmıyor; karara etki etmedi.",
        sample_explanation,
    )
    assert not audit.protected_violations


def test_detects_missing_what_if_tool_call(sample_explanation) -> None:
    """Varsayımsal soruda tool çağrılmadan cevap verilmesi ihlaldir."""
    audit = audit_narrative(
        "Vade düşse risk azalır ve büyük olasılıkla onaylanırdınız.",
        sample_explanation,
        question="Kredi vadesi 12 aya düşse ne olurdu?",
        used_tools=["get_decision_explanation"],
    )
    assert audit.missing_tool_call


def test_what_if_tool_call_satisfies_requirement(sample_explanation) -> None:
    audit = audit_narrative(
        "Model yeniden koştu ve karar değişmedi.",
        sample_explanation,
        question="Vade 12 aya düşse ne olurdu?",
        used_tools=["run_what_if"],
    )
    assert not audit.missing_tool_call


def test_report_aggregation(sample_explanation) -> None:
    from xai_agent.faithfulness import NarrativeFaithfulnessReport

    clean = _audit("Karar açıklandı.", sample_explanation)
    dirty = _audit("Geliriniz yetersiz.", sample_explanation)
    report = NarrativeFaithfulnessReport([clean, dirty])
    assert report.n == 2
    assert report.n_passed == 1
    assert report.score == 0.5
    payload = report.to_dict()
    assert payload["violations_by_type"]["fabricated_concepts"] >= 1
    assert "misframed_shares" in payload["violations_by_type"]


# --------------------------------------------------------------------------
# SHAP sadakati (ERASER metrikleri)
# --------------------------------------------------------------------------
def test_reference_row_uses_median_and_mode(dataset) -> None:
    ref = reference_row(dataset.X_train)
    assert set(ref) == set(dataset.X_train.columns)
    assert ref["age"] == dataset.X_train["age"].median()
    assert ref["housing"] == dataset.X_train["housing"].mode().iloc[0]


def test_shap_beats_random_baseline(explainer, dataset) -> None:
    """SHAP sıralaması rastgele özellik seçimini yenmeli.

    Yenmiyorsa "en önemli özellikler" sıralamasının hiçbir bilgi değeri yok
    demektir ve açıklamalar dekoratif olur.
    """
    result = evaluate_shap_faithfulness(
        explainer,
        X_eval=dataset.X_test,
        X_train=dataset.X_train,
        k_values=(3, 5),
        n_samples=80,
    )
    assert result.additivity["passed"]
    # Eşik ölçümle belirlendi: n=40..200 aralığında lift 1.57–2.05 arasında
    # dalgalanıyor (bkz. README "Sonuçlar"). 1.4 güvenli bir alt sınır.
    assert result.lift_over_random > 1.4, (
        f"SHAP rastgeleyi yenemedi: {result.lift_over_random:.2f}x"
    )
    assert result.comprehensiveness[5] > result.random_comprehensiveness[5] * 1.3


def test_comprehensiveness_increases_with_k(explainer, dataset) -> None:
    """Daha çok önemli özellik kaldırıldıkça tahmin daha çok düşmeli."""
    result = evaluate_shap_faithfulness(
        explainer,
        X_eval=dataset.X_test,
        X_train=dataset.X_train,
        k_values=(1, 3, 5),
        n_samples=80,
    )
    values = [result.comprehensiveness[k] for k in (1, 3, 5)]
    assert values[0] < values[1] < values[2], f"monoton değil: {values}"


def test_sufficiency_decreases_with_k(explainer, dataset) -> None:
    """Daha çok önemli özellik bırakıldıkça orijinalden sapma azalmalı."""
    result = evaluate_shap_faithfulness(
        explainer,
        X_eval=dataset.X_test,
        X_train=dataset.X_train,
        k_values=(1, 3, 5),
        n_samples=80,
    )
    values = [result.sufficiency[k] for k in (1, 3, 5)]
    assert values[0] > values[1] > values[2], f"monoton değil: {values}"
