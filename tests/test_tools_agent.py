"""Faz 3 testleri: tool'lar, adalet denetimi ve (isteğe bağlı) canlı ajan.

Tool testleri LLM gerektirmez — fonksiyonları doğrudan çağırıyoruz. Canlı
ajan testleri ``@pytest.mark.llm`` ile işaretli ve Ollama kapalıysa atlanır;
böylece CI'da veya çevrimdışı çalışırken test paketi kırılmaz.
"""

from __future__ import annotations

import pytest

from xai_agent.tools import AgentToolbox, _coerce_value


# --------------------------------------------------------------------------
# Değer dönüştürme — LLM her zaman string gönderir
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feature,raw,expected",
    [
        ("duration", "24", 24),
        ("duration", "24 ay", 24),
        ("credit_amount", "3000", 3000),
        ("age", "35", 35),
        ("checking_status", ">=200", ">=200"),
        ("checking_status", "bakiye 200 DM ve üzeri", ">=200"),
        ("housing", "own", "own"),
        ("housing", "ev sahibi", "own"),
    ],
)
def test_coerce_value_accepts_valid_inputs(feature, raw, expected) -> None:
    assert _coerce_value(feature, raw) == expected


@pytest.mark.parametrize(
    "feature,raw,pattern",
    [
        ("duration", "çok uzun", "sayıya çevrilemedi"),
        ("duration", "500", "aralığın dışında"),
        ("age", "5", "aralığın dışında"),
        ("housing", "sarayda", "geçerli bir değer değil"),
        ("gelir", "5000", "bilinen bir özellik değil"),
    ],
)
def test_coerce_value_rejects_invalid_inputs(feature, raw, pattern) -> None:
    with pytest.raises(ValueError, match=pattern):
        _coerce_value(feature, raw)


def test_coerce_value_guards_training_range() -> None:
    """Veri setinin dışındaki değerler reddedilmeli.

    Model 19-75 yaş aralığında eğitildi. 200 yaşında bir başvuru için
    yaptığı tahmin anlamsızdır (ekstrapolasyon). Sessizce bir sayı
    döndürmek yerine hata veriyoruz.
    """
    with pytest.raises(ValueError, match="güvenilir tahmin üretmez"):
        _coerce_value("age", "200")


# --------------------------------------------------------------------------
# Toolbox
# --------------------------------------------------------------------------
@pytest.fixture
def toolbox(explainer, sample_applicant) -> AgentToolbox:
    return AgentToolbox(
        explainer=explainer, applicant=dict(sample_applicant), applicant_id="T-000"
    )


def test_toolbox_exposes_four_tools(toolbox) -> None:
    names = {t.__name__ for t in toolbox.build_tools()}
    assert names == {
        "get_decision_explanation",
        "run_what_if",
        "get_feature_info",
        "get_global_importance",
    }


def test_toolbox_rejects_incomplete_applicant(explainer, sample_applicant) -> None:
    incomplete = dict(sample_applicant)
    incomplete.pop("duration")
    with pytest.raises(ValueError, match="eksik özellik"):
        AgentToolbox(explainer=explainer, applicant=incomplete)


def test_get_decision_explanation_returns_payload(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    payload = tools["get_decision_explanation"]()
    assert payload["karar"] in ("onaylandı", "reddedildi")
    assert payload["risk_orani"].startswith("%")
    assert payload["riski_artiran_etkenler"]
    assert toolbox.call_log[-1]["tool"] == "get_decision_explanation"


def test_run_what_if_returns_real_recomputation(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    result = tools["run_what_if"]("duration", "72")
    assert "hata" not in result
    assert result["onceki_risk_orani"] != result["yeni_risk_orani"]
    assert result["karar_degisti_mi"] in ("EVET", "HAYIR")


def test_run_what_if_returns_error_dict_not_exception(toolbox) -> None:
    """Hatalı tool çağrısı istisna atmamalı — modele anlamlı mesaj dönmeli.

    Ajan yanlış parametreyle çağırdığında istisna atarsak tüm tur çöker.
    Bunun yerine hata mesajını modele geri veriyoruz; model düzeltip tekrar
    deneyebiliyor ya da kullanıcıya durumu açıklayabiliyor.
    """
    tools = {t.__name__: t for t in toolbox.build_tools()}
    result = tools["run_what_if"]("housing", "sarayda")
    assert "hata" in result
    assert "UYDURMA" in result["yapilacak"].upper()
    assert toolbox.call_log[-1]["ok"] is False


def test_run_what_if_rejects_protected_feature(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    result = tools["run_what_if"]("personal_status", "male single")
    assert "hata" in result


def test_get_feature_info_categorical(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    info = tools["get_feature_info"]("checking_status")
    assert info["tip"] == "kategorik"
    assert len(info["gecerli_degerler"]) == 4
    assert "bu_basvurudaki_degeri" in info


def test_get_feature_info_numeric(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    info = tools["get_feature_info"]("duration")
    assert info["tip"] == "sayısal"
    assert info["gecerli_aralik"] == "4 – 72"
    assert info["birim"] == "ay"


def test_get_feature_info_unknown_returns_help(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    info = tools["get_feature_info"]("gelir")
    assert "hata" in info
    assert "gecerli_ozellikler" in info


def test_get_global_importance(toolbox) -> None:
    tools = {t.__name__: t for t in toolbox.build_tools()}
    result = tools["get_global_importance"](5)
    if "hata" in result:
        pytest.skip("Küresel önem önbelleği yok (scripts/train.py çalıştırın)")
    assert len(result["ozellikler"]) == 5
    assert result["ozellikler"][0]["sira"] == 1
    assert result["ozellikler"][0]["genel_agirligi"].startswith("%")
    assert "DEĞİLDİR" in result["aciklama"]


def test_toolbox_reset_clears_cache(toolbox, dataset) -> None:
    tools = {tool.__name__: tool for tool in toolbox.build_tools()}
    tools["get_decision_explanation"]()
    assert toolbox.call_log, "çağrı kaydı tutulmalı"

    toolbox.reset(dataset.X_test.iloc[10].to_dict(), "T-010")
    assert toolbox.call_log == [], "reset çağrı kaydını temizlemeli"
    assert toolbox.applicant_id == "T-010"
    assert toolbox.explanation.applicant_id == "T-010", (
        "önbelleklenmiş açıklama yeni başvuru için yeniden hesaplanmalı"
    )


# --------------------------------------------------------------------------
# Adalet denetimi
# --------------------------------------------------------------------------
def test_fairness_audit_runs(bundle) -> None:
    from xai_agent.fairness import run_fairness_audit

    report = run_fairness_audit(bundle, save=False)
    assert report["n_test"] == 200
    assert set(report["excluded_from_model"]) == {
        "personal_status",
        "foreign_worker",
    }
    assert report["summary"], "Özet boş olmamalı"
    attributes = {item["attribute"] for item in report["summary"]}
    assert "Yaş grubu" in attributes


def test_fairness_gap_interpretation() -> None:
    from xai_agent.fairness import interpret_gap

    assert "ihmal" in interpret_gap(0.01)
    assert "küçük" in interpret_gap(0.07)
    assert "DİKKAT" in interpret_gap(0.15)
    assert "CİDDİ" in interpret_gap(0.30)
    assert "hesaplanamadı" in interpret_gap(None)


def test_fairness_flags_small_groups_as_unreliable(bundle) -> None:
    """20'den az kişilik gruplar 'güvenilmez' işaretlenmeli.

    ``foreign_worker`` veri setinde çok dengesiz (%96/%4). Küçük gruptan
    metrik üretip "ayrımcılık var/yok" demek istatistiksel olarak yanlış olur.
    """
    from xai_agent.fairness import run_fairness_audit

    report = run_fairness_audit(bundle, save=False)
    detail = next(
        d for d in report["details"] if d["attribute"] == "foreign_worker"
    )
    assert any(not g["reliable"] for g in detail["groups"])


# --------------------------------------------------------------------------
# LLM istemcisi
# --------------------------------------------------------------------------
def test_backend_validation_rejects_unknown() -> None:
    from xai_agent.config import LLMSettings
    from xai_agent.llm import LLMBackendError, create_chat_client

    settings = LLMSettings(llm_backend="kesinlikle-olmayan-backend")
    with pytest.raises(LLMBackendError, match="Bilinmeyen arka uç"):
        create_chat_client(settings)


def test_describe_backend() -> None:
    from xai_agent.llm import describe_backend

    info = describe_backend()
    assert info["backend"] in ("ollama", "foundry", "azure")
    assert info["model"]


# --------------------------------------------------------------------------
# Onarım döngüsü (critic-and-revise) — LLM gerekmeyen kısım
# --------------------------------------------------------------------------
def test_repair_prompt_lists_every_violation_type() -> None:
    """Onarım mesajı her ihlal türünü ajana somut biçimde bildirmeli."""
    from xai_agent.prompts import build_repair_prompt

    class FakeAudit:
        ungrounded_numbers = [99.7]
        fabricated_concepts = ["gelir"]
        direction_conflicts = [
            {"feature": "duration", "shap_direction": "riski_artiriyor"}
        ]
        misframed_shares = [{"value": "%21.3"}]
        protected_violations = ["Kadın olmanız nedeniyle"]
        missing_tool_call = True

    message = build_repair_prompt(FakeAudit())
    assert "99.7" in message
    assert "gelir" in message
    assert "duration" in message
    assert "riski ARTIRIYOR" in message
    assert "%21.3" in message
    assert "run_what_if" in message
    # Ajana "özür dileme, sadece düzelt" talimatı verilmeli
    assert "BAŞTAN yaz" in message


def test_repair_prompt_empty_when_clean() -> None:
    """İhlal yoksa onarım mesajı üretilmemeli (gereksiz LLM çağrısı olmasın)."""
    from xai_agent.prompts import build_repair_prompt

    class CleanAudit:
        ungrounded_numbers: list = []
        fabricated_concepts: list = []
        direction_conflicts: list = []
        misframed_shares: list = []
        protected_violations: list = []
        missing_tool_call = False

    assert build_repair_prompt(CleanAudit()) == ""


def test_verified_answer_reports_improvement() -> None:
    """VerifiedAnswer, onarımın gerçekten iyileştirip iyileştirmediğini bildirmeli."""
    from xai_agent.agent import AgentTurn, VerifiedAnswer

    class Audit:
        def __init__(self, violations: int):
            self.violations = violations
            self.passed = violations == 0

    turn = AgentTurn(question="s", answer="c")
    improved = VerifiedAnswer(
        question="s",
        turn=turn,
        audit=Audit(0),
        first_audit=Audit(3),
        repair_rounds=1,
        first_attempt_violations=3,
        attempts=2,
    )
    assert improved.passed
    assert improved.was_repaired
    assert improved.improved
    assert improved.summary()["first_attempt_violations"] == 3

    unchanged = VerifiedAnswer(
        question="s",
        turn=turn,
        audit=Audit(2),
        first_audit=Audit(2),
        repair_rounds=1,
        first_attempt_violations=2,
        attempts=2,
    )
    assert not unchanged.improved
    assert not unchanged.passed


# --------------------------------------------------------------------------
# Canlı ajan (LLM gerekir)
# --------------------------------------------------------------------------
@pytest.mark.llm
def test_agent_calls_tool_for_explanation(
    ollama_available, explainer, sample_applicant
) -> None:
    """Ajan açıklama isteğinde tool çağırmak ZORUNDA."""
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor")
    from xai_agent.agent import CreditAgent

    agent = CreditAgent(explainer)
    agent.set_applicant(sample_applicant, "T-000")
    turn = agent.ask_turn("Bu başvuru neden bu sonucu aldı?")
    assert turn.answer, "Ajan boş yanıt döndürdü"
    assert "get_decision_explanation" in turn.tool_names, (
        f"Tool çağrılmadı! çağrılanlar: {turn.tool_names}"
    )


@pytest.mark.llm
def test_agent_calls_what_if_for_hypothetical(
    ollama_available, explainer, sample_applicant
) -> None:
    """Varsayımsal soruda ajan modeli yeniden koşturmalı, tahmin yürütmemeli."""
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor")
    from xai_agent.agent import CreditAgent

    agent = CreditAgent(explainer)
    agent.set_applicant(sample_applicant, "T-000")
    turn = agent.ask_turn("Kredi vadesi 6 aya düşse karar değişir miydi?")
    assert "run_what_if" in turn.tool_names, (
        f"Varsayımsal soruda what-if çağrılmadı! çağrılanlar: {turn.tool_names}"
    )


@pytest.mark.llm
def test_agent_answer_is_faithful(
    ollama_available, explainer, sample_applicant
) -> None:
    """Ajanın açıklaması sadakat denetimini geçmeli."""
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor")
    from xai_agent.agent import CreditAgent
    from xai_agent.faithfulness import audit_narrative

    agent = CreditAgent(explainer)
    agent.set_applicant(sample_applicant, "T-000")
    turn = agent.explain_decision()
    audit = audit_narrative(
        turn.answer,
        agent.explanation,
        question="açıklama",
        used_tools=turn.tool_names,
    )
    assert not audit.fabricated_concepts, (
        f"Ajan uydurma kavram kullandı: {audit.fabricated_concepts}"
    )
    assert not audit.protected_violations, (
        f"Korunan özellik ihlali: {audit.protected_violations}"
    )
    assert not audit.ungrounded_numbers, (
        f"Temellenmemiş sayılar: {audit.ungrounded_numbers}"
    )


@pytest.mark.llm
def test_agent_ask_verified_never_returns_worse_answer(
    ollama_available, explainer, sample_applicant
) -> None:
    """Onarım döngüsü sonucu asla ilk denemeden KÖTÜ olmamalı.

    ``ask_verified`` tüm denemeler arasından en az ihlalliyi döndürür; düzeltme
    girişimi durumu bozarsa ilk yanıta geri dönülür.
    """
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor")
    from xai_agent.agent import CreditAgent

    agent = CreditAgent(explainer)
    agent.set_applicant(sample_applicant, "T-000")
    verified = agent.ask_verified(
        "Bu başvurunun sonucunu ve nedenlerini açıkla.", max_repairs=1
    )
    assert verified.answer
    assert verified.audit.violations <= verified.first_attempt_violations


@pytest.mark.llm
def test_agent_resets_history_on_new_applicant(
    ollama_available, explainer, dataset
) -> None:
    """Başvuru değişince sohbet geçmişi sıfırlanmalı — sayı taşınmasın."""
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor")
    from xai_agent.agent import CreditAgent

    agent = CreditAgent(explainer)
    agent.set_applicant(dataset.X_test.iloc[0].to_dict(), "T-000")
    agent.ask_turn("Kararı açıkla.")
    assert len(agent.history) == 1
    agent.set_applicant(dataset.X_test.iloc[1].to_dict(), "T-001")
    assert agent.history == []
    assert agent.toolbox.applicant_id == "T-001"
