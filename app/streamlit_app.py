"""XAI-Agent Streamlit arayüzü — üç katmanın buluştuğu yer.

Çalıştırma::

    uv run streamlit run app/streamlit_app.py

Tasarım ilkesi
--------------
Arayüz, ajanın anlatısını **SHAP grafiğinin yanında** gösterir. Sebep şu:
kullanıcı ajanın sözünü grafiğe bakarak doğrulayabilmeli. Açıklanabilirlik
iddiası olan bir sistemde LLM metnini tek başına sunmak, kullanıcıyı ikinci
bir kara kutuya mahkûm etmek olur.

Ayrıca her ekranda ajanın hangi tool'ları çağırdığı gösterilir. Tool
çağrılmadan üretilmiş bir cevap, tanım gereği temellenmemiş bir cevaptır ve
kullanıcı bunu görebilmeli.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from xai_agent import config  # noqa: E402
from xai_agent.data import prepare  # noqa: E402
from xai_agent.explainer import CreditExplainer  # noqa: E402
from xai_agent.faithfulness import audit_narrative  # noqa: E402
from xai_agent.features import (  # noqa: E402
    ACTIONABLE_FEATURES,
    FEATURE_META,
    display_name,
)
from xai_agent.llm import describe_backend  # noqa: E402
from xai_agent.model import ModelBundle  # noqa: E402

st.set_page_config(
    page_title="XAI-Agent · Şeffaf Kredi Kararı",
    page_icon="🔍",
    layout="wide",
)


# --------------------------------------------------------------------------
# Önbelleklenmiş kaynaklar
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_bundle() -> ModelBundle:
    return ModelBundle.load()


@st.cache_resource(show_spinner="SHAP açıklayıcı hazırlanıyor...")
def load_explainer() -> CreditExplainer:
    return CreditExplainer(load_bundle())


@st.cache_data(show_spinner="Veri hazırlanıyor...")
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    ds = prepare()
    return ds.X_test, ds.y_test


def get_agent():
    """Ajanı oturum durumunda tutar (her etkileşimde yeniden kurulmasın)."""
    if "agent" not in st.session_state:
        from xai_agent.agent import CreditAgent

        st.session_state.agent = CreditAgent(load_explainer())
    return st.session_state.agent


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------
def build_manual_form(defaults: dict) -> dict:
    """Özellik sözlüğünden manuel giriş formu üretir."""
    values: dict = {}
    bundle = load_bundle()
    cols = st.columns(3)
    for i, name in enumerate(bundle.feature_names):
        meta = FEATURE_META[name]
        col = cols[i % 3]
        with col:
            if meta.kind == "categorical":
                options = list(meta.levels)
                current = str(defaults.get(name, options[0]))
                index = options.index(current) if current in options else 0
                values[name] = st.selectbox(
                    meta.display,
                    options,
                    index=index,
                    format_func=lambda v, m=meta: m.value_labels.get(v, v),
                    key=f"manual_{name}",
                    help=meta.description,
                )
            else:
                lo, hi, step = meta.bounds or (0, 100, 1)
                values[name] = st.number_input(
                    f"{meta.display}"
                    + (f" ({meta.unit})" if meta.unit else ""),
                    min_value=int(lo),
                    max_value=int(hi),
                    value=int(defaults.get(name, (lo + hi) // 2)),
                    step=int(step),
                    key=f"manual_{name}",
                    help=meta.description,
                )
    return values


def decision_card(explanation) -> None:
    """Kararı üst bantta özetler."""
    rejected = explanation.decision == "reddedildi"
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.6])
    with c1:
        if rejected:
            st.error(f"### {explanation.decision.upper()}")
        else:
            st.success(f"### {explanation.decision.upper()}")
    c2.metric("Risk oranı", f"{explanation.risk_probability:.1%}")
    c3.metric(
        "Karar eşiği",
        f"{explanation.threshold:.1%}",
        delta=f"{explanation.margin:+.1%}",
        delta_color="inverse",
    )
    band = explanation.confidence_band.value
    with c4:
        if band == "sinirda":
            st.warning("**KIL PAYI** — eşiğe çok yakın, küçük bir değişiklik "
                       "sonucu çevirebilir")
        elif band == "orta":
            st.info("**Orta netlikte** — eşiğe makul mesafe var")
        else:
            st.info("**Net karar** — eşikten belirgin biçimde uzak")


def contributions_table(explanation) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Sıra": c.rank,
                "Özellik": c.display_name,
                "Başvurudaki değer": c.value_label,
                "Yön": (
                    "▲ riski artırıyor"
                    if c.direction.value == "riski_artiriyor"
                    else "▼ riski azaltıyor"
                ),
                "SHAP (log-odds)": round(c.shap_value, 4),
                "Etki payı": f"%{c.share_of_total:.1f}",
                "Değiştirilebilir": "evet" if c.actionable else "hayır",
            }
            for c in explanation.all_contributions
        ]
    )


# --------------------------------------------------------------------------
# Kenar çubuğu
# --------------------------------------------------------------------------
def sidebar() -> tuple[dict, str]:
    st.sidebar.title("🔍 XAI-Agent")
    st.sidebar.caption(
        "LightGBM → SHAP → LLM ajanı. Karar modelin, gerekçe SHAP'ın, "
        "dil ajanın işi."
    )

    bundle = load_bundle()
    X_test, y_test = load_test_data()

    st.sidebar.divider()
    st.sidebar.subheader("Başvuru seçimi")
    mode = st.sidebar.radio(
        "Kaynak",
        ["Test setinden seç", "Manuel giriş"],
        label_visibility="collapsed",
    )

    if mode == "Test setinden seç":
        scores = bundle.predict_proba(X_test)
        filter_choice = st.sidebar.selectbox(
            "Filtre",
            ["Tümü", "Yalnızca reddedilenler", "Yalnızca onaylananlar",
             "Eşiğe en yakınlar (kıl payı)"],
        )
        idx_pool = list(range(len(X_test)))
        if filter_choice == "Yalnızca reddedilenler":
            idx_pool = [i for i in idx_pool if scores[i] >= bundle.threshold]
        elif filter_choice == "Yalnızca onaylananlar":
            idx_pool = [i for i in idx_pool if scores[i] < bundle.threshold]
        elif filter_choice == "Eşiğe en yakınlar (kıl payı)":
            idx_pool = sorted(
                idx_pool, key=lambda i: abs(scores[i] - bundle.threshold)
            )[:25]

        if not idx_pool:
            st.sidebar.warning("Bu filtreye uyan başvuru yok.")
            idx_pool = list(range(len(X_test)))

        chosen = st.sidebar.selectbox(
            "Başvuru",
            idx_pool,
            format_func=lambda i: (
                f"A-{i:03d} · risk {scores[i]:.0%} · "
                f"{'RED' if scores[i] >= bundle.threshold else 'ONAY'}"
                f" · gerçek: {'riskli' if y_test.iloc[i] == 1 else 'iyi'}"
            ),
        )
        applicant = X_test.iloc[int(chosen)].to_dict()
        applicant_id = f"A-{int(chosen):03d}"
    else:
        st.sidebar.info("Değerleri ana ekrandaki formdan girin.")
        applicant = st.session_state.get(
            "manual_applicant", X_test.iloc[0].to_dict()
        )
        applicant_id = "MANUEL"

    st.sidebar.divider()
    st.sidebar.subheader("LLM arka ucu")
    backend = describe_backend()
    st.sidebar.code(
        f"backend : {backend['backend']}\n"
        f"model   : {backend['model']}\n"
        f"adres   : {backend['base_url']}",
        language="text",
    )
    if st.sidebar.button("Bağlantıyı sına", width="stretch"):
        ok, msg = get_agent().health_check()
        (st.sidebar.success if ok else st.sidebar.error)(msg)

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    test_metrics = bundle.metrics.get("test", {})
    st.sidebar.code(
        f"aile      : LightGBM\n"
        f"ağaç       : {bundle.n_estimators_used}\n"
        f"özellik    : {len(bundle.feature_names)}\n"
        f"ROC-AUC    : {test_metrics.get('roc_auc', '—')}\n"
        f"PR-AUC     : {test_metrics.get('pr_auc', '—')}\n"
        f"eşik       : {bundle.threshold}",
        language="text",
    )
    st.sidebar.caption(
        "Cinsiyet, medeni durum ve uyruk bilgisi modelden bilinçli olarak "
        "çıkarılmıştır."
    )
    return applicant, applicant_id, mode


# --------------------------------------------------------------------------
# Sekmeler
# --------------------------------------------------------------------------
def tab_explanation(agent, explanation) -> None:
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Ajanın açıklaması")
        st.caption(
            "Bu metin, yandaki SHAP grafiğindeki sayılardan üretilir. "
            "Ajanın tahmin yapma yetkisi yoktur."
        )

        repair = st.toggle(
            "🛡️ Denetimli mod (critic-and-revise)",
            value=True,
            help=(
                "Yanıt sadakat denetiminden geçemezse ihlaller ajana geri "
                "bildirilir ve yeniden yazması istenir. Ölçümler bu modun "
                "ihlalleri belirgin biçimde azalttığını gösteriyor; karşılığı "
                "daha uzun yanıt süresi."
            ),
        )

        if st.button("🗣️ Açıklamayı üret", type="primary", width="stretch"):
            ok, msg = agent.health_check()
            if not ok:
                st.error(msg)
            else:
                spinner = (
                    "Ajan açıklamayı yazıyor, ardından denetimden geçirilecek..."
                    if repair
                    else "Ajan SHAP çıktısını okuyup açıklama yazıyor..."
                )
                with st.spinner(spinner):
                    try:
                        question = (
                            "Bu başvurunun sonucunu ve nedenlerini başvuru "
                            "sahibine açıkla."
                        )
                        if repair:
                            verified = agent.ask_verified(question, max_repairs=1)
                            st.session_state.first_turn = verified.turn
                            st.session_state.first_verified = verified
                        else:
                            st.session_state.first_turn = agent.ask_turn(question)
                            st.session_state.first_verified = None
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Ajan yanıt üretemedi: {exc}")

        turn = st.session_state.get("first_turn")
        if turn:
            st.markdown(turn.answer)
            with st.expander("🔧 Ajan hangi tool'ları çağırdı?", expanded=False):
                if turn.tool_calls:
                    for c in turn.tool_calls:
                        status = "✅" if c["ok"] else f"❌ {c['note']}"
                        st.code(f"{c['tool']}({c['args']})  {status}", language="text")
                else:
                    st.warning(
                        "Tool çağrılmadı! Bu yanıt temellenmemiş olabilir."
                    )

            verified = st.session_state.get("first_verified")
            if verified is not None:
                audit = verified.audit
                if verified.was_repaired:
                    st.caption(
                        f"🛡️ Denetimli mod devreye girdi: ilk yanıtta "
                        f"{verified.first_attempt_violations} ihlal bulundu, "
                        f"düzeltme sonrası {audit.violations}."
                    )
            else:
                audit = audit_narrative(
                    turn.answer, explanation,
                    question="açıklama", used_tools=turn.tool_names,
                )

            if audit.passed:
                st.success(
                    f"✅ Sadakat denetimi geçti — {audit.grounded_number_count} "
                    "sayının tamamı SHAP çıktısıyla temellendirildi."
                )
            else:
                st.error(
                    f"⚠️ Sadakat denetimi {audit.violations} ihlal buldu. "
                    "Bu uyarı bilinçli olarak kullanıcıya gösteriliyor: "
                    "denetlenemeyen bir açıklama, açıklama değildir."
                )
                st.json(audit.to_dict(), expanded=False)
        else:
            st.info("Açıklamayı üretmek için yukarıdaki düğmeye basın.")

    with right:
        st.subheader("SHAP şelale grafiği")
        st.caption(
            "Taban değerden başlayıp her özelliğin katkısıyla nihai karara "
            "ulaşan yol. Kırmızı = riski artırıyor, yeşil = azaltıyor."
        )
        fig = load_explainer().waterfall_figure(explanation, top_k=10)
        st.pyplot(fig, width="stretch")
        a = explanation.additivity
        st.caption(
            f"Matematiksel doğrulama: taban ({a.base_value_logodds:+.4f}) + "
            f"katkılar ({a.sum_shap:+.4f}) = {a.reconstructed_logodds:+.4f} ≈ "
            f"modelin çıktısı ({a.model_logodds:+.4f}). "
            f"Hata: {a.abs_error:.1e} → "
            f"{'GEÇTİ ✅' if a.passed else 'BAŞARISIZ ❌'}"
        )


def tab_factors(explanation) -> None:
    st.subheader("Tüm etkenler")
    st.caption(
        "18 özelliğin tamamı, etkisinin büyüklüğüne göre sıralı. Ajan yalnızca "
        f"ilk {config.TOP_K_DRIVERS}+{config.TOP_K_DRIVERS} etkeni görür; "
        "buradaki tablo tam dökümdür."
    )
    df = contributions_table(explanation)
    st.dataframe(df, width="stretch", hide_index=True, height=560)

    c1, c2 = st.columns(2)
    c1.metric(
        "Gösterilen etkenlerin kapsamı", f"%{explanation.coverage_percent:.0f}"
    )
    c2.metric(
        "Riski artıran / azaltan",
        f"{len(explanation.top_risk_drivers)} / "
        f"{len(explanation.top_protective_factors)}",
    )
    if explanation.fairness_note:
        st.info(explanation.fairness_note)


def tab_what_if(agent, applicant: dict, explanation) -> None:
    st.subheader("What-if — karşı-olgusal senaryo")
    st.caption(
        "Bir özelliği değiştirip modeli **gerçekten** yeniden koşar. "
        "Sonuçlar LLM tahmini değil, LightGBM'in yeni çıktısıdır."
    )

    actionable = [
        f for f in ACTIONABLE_FEATURES if f in agent.explainer.bundle.feature_names
    ]
    c1, c2 = st.columns(2)
    feature = c1.selectbox(
        "Değiştirilecek özellik",
        actionable,
        format_func=display_name,
        help="Yalnızca başvuru sahibinin etkileyebileceği özellikler listelenir.",
    )
    meta = FEATURE_META[feature]
    current = applicant[feature]
    c1.caption(f"Şu anki değer: **{meta.label_for(current)}**")

    if meta.kind == "categorical":
        options = [v for v in meta.levels if str(v) != str(current)]
        new_value = c2.selectbox(
            "Yeni değer",
            options,
            format_func=lambda v: meta.value_labels.get(v, v),
        )
    else:
        lo, hi, step = meta.bounds or (0, 100, 1)
        new_value = c2.slider(
            f"Yeni değer ({meta.unit or ''})",
            min_value=int(lo),
            max_value=int(hi),
            value=int(current),
            step=int(step),
        )

    if st.button("▶️ Senaryoyu çalıştır", type="primary"):
        try:
            result = agent.explainer.what_if(applicant, {feature: new_value})
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state.what_if = result

    result = st.session_state.get("what_if")
    if result:
        d1, d2, d3 = st.columns(3)
        d1.metric("Önceki risk", f"{result.baseline_probability:.1%}")
        d2.metric(
            "Yeni risk",
            f"{result.new_probability:.1%}",
            delta=f"{result.delta_probability:+.1%}",
            delta_color="inverse",
        )
        d3.metric(
            "Karar",
            result.new_decision.upper(),
            delta="DEĞİŞTİ" if result.decision_flipped else "değişmedi",
            delta_color="normal" if result.decision_flipped else "off",
        )
        if result.decision_flipped:
            st.success(f"Karar değişti: {result.baseline_decision} → "
                       f"{result.new_decision}")
        else:
            st.warning(result.note)

        if result.top_changed_contributions:
            st.markdown("**En çok değişen katkılar**")
            st.dataframe(
                pd.DataFrame(result.top_changed_contributions),
                width="stretch",
                hide_index=True,
            )


def tab_chat(agent, explanation) -> None:
    st.subheader("Ajanla konuş")
    st.caption(
        "Takip sorusu sorun. Varsayımsal sorularda ajan `run_what_if` tool'unu "
        "çağırmak zorundadır — çağırmazsa aşağıda uyarı görürsünüz."
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("repair_note"):
                st.caption(msg["repair_note"])
            if msg.get("tools") is not None:
                if msg["tools"]:
                    st.caption(f"🔧 çağrılan tool'lar: {', '.join(msg['tools'])}")
                else:
                    st.caption("⚠️ tool çağrılmadı — bu yanıt temellenmemiş olabilir")
            if msg.get("audit"):
                a = msg["audit"]
                if a["passed"]:
                    st.caption(f"✅ sadakat denetimi geçti "
                               f"({a['grounded_number_count']} temelli sayı)")
                else:
                    st.caption(f"⚠️ sadakat denetimi {a['violations']} ihlal buldu")
                    with st.expander("ihlal detayı"):
                        st.json(a)

    repair = st.toggle(
        "🛡️ Denetimli mod",
        value=True,
        key="chat_repair",
        help="İhlalli yanıtlar ajana geri gönderilip düzeltilir.",
    )

    question = st.chat_input("Örn: Vade 24 aya düşse karar değişir miydi?")
    if not question:
        return

    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        ok, msg = agent.health_check()
        if not ok:
            st.error(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
            return
        with st.spinner("Ajan tool'ları çağırıyor..."):
            try:
                if repair:
                    verified = agent.ask_verified(question, max_repairs=1)
                    turn, audit = verified.turn, verified.audit
                    repair_note = (
                        f"🛡️ onarım: {verified.first_attempt_violations} ihlal "
                        f"-> {audit.violations}"
                        if verified.was_repaired
                        else None
                    )
                else:
                    turn = agent.ask_turn(question)
                    audit = audit_narrative(
                        turn.answer, explanation, question=question,
                        used_tools=turn.tool_names,
                    )
                    repair_note = None
            except Exception as exc:  # noqa: BLE001
                err = f"Ajan yanıt üretemedi: {exc}"
                st.error(err)
                st.session_state.chat.append({"role": "assistant", "content": err})
                return
        st.markdown(turn.answer)
        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": turn.answer,
                "tools": turn.tool_names,
                "audit": audit.to_dict(),
                "repair_note": repair_note,
            }
        )
        st.rerun()


def tab_audit() -> None:
    st.subheader("Model ve denetim raporları")
    st.caption(
        "Bu projenin iddiası ölçülebilir olmasıdır. Aşağıdaki raporlar "
        "`scripts/evaluate.py` tarafından üretilir."
    )
    import json

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Sadakat (faithfulness)")
        if config.FAITHFULNESS_PATH.exists():
            data = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))
            shap_part = data.get("shap_faithfulness", {})
            st.metric("SHAP kazancı (rastgeleye karşı)",
                      f"{shap_part.get('lift_over_random', '—')}x")
            st.write(shap_part.get("verdict", ""))
            narrative = data.get("narrative_faithfulness")
            if narrative:
                st.metric("Ajan sadakat skoru",
                          f"{narrative['faithfulness_score']:.0%}")
                st.write(narrative["verdict"])
            st.json(data, expanded=False)
        else:
            st.info("Henüz rapor üretilmedi:  uv run python scripts/evaluate.py")

    with c2:
        st.markdown("#### Adalet (fairness)")
        if config.FAIRNESS_PATH.exists():
            data = json.loads(config.FAIRNESS_PATH.read_text(encoding="utf-8"))
            st.dataframe(
                pd.DataFrame(data["summary"]), width="stretch", hide_index=True
            )
            st.caption(data["impossibility_note"])
            st.json(data, expanded=False)
        else:
            st.info("Henüz rapor üretilmedi:  uv run python scripts/evaluate.py")

    st.divider()
    st.markdown("#### Metrikler")
    if config.METRICS_PATH.exists():
        st.json(
            json.loads(config.METRICS_PATH.read_text(encoding="utf-8")),
            expanded=False,
        )


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------
def main() -> None:
    try:
        load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    applicant, applicant_id, mode = sidebar()

    st.title("Şeffaf Kredi Kararı")
    st.caption(
        "LightGBM tahmin eder · SHAP gerekçeyi sayısallaştırır · "
        "LLM ajanı doğal dile çevirir. Her katman denetlenebilir."
    )

    if mode == "Manuel giriş":
        with st.expander("📝 Başvuru bilgilerini girin", expanded=True):
            applicant = build_manual_form(applicant)
            st.session_state.manual_applicant = applicant

    explainer = load_explainer()
    explanation = explainer.explain_row(applicant, applicant_id=applicant_id)

    agent = get_agent()
    signature = (applicant_id, tuple(sorted(map(str, applicant.items()))))
    if st.session_state.get("signature") != signature:
        agent.set_applicant(applicant, applicant_id)
        st.session_state.signature = signature
        st.session_state.pop("first_turn", None)
        st.session_state.pop("first_verified", None)
        st.session_state.pop("what_if", None)
        st.session_state.chat = []

    decision_card(explanation)
    st.divider()

    t1, t2, t3, t4, t5 = st.tabs(
        ["🗣️ Açıklama", "📊 Etkenler", "🔀 What-if", "💬 Sohbet", "🧪 Denetim"]
    )
    with t1:
        tab_explanation(agent, explanation)
    with t2:
        tab_factors(explanation)
    with t3:
        tab_what_if(agent, applicant, explanation)
    with t4:
        tab_chat(agent, explanation)
    with t5:
        tab_audit()


main()
