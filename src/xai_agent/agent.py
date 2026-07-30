"""Ajan katmanı: SHAP çıktısını doğal dile çeviren LLM sarmalayıcısı.

Bu katmanın tek işi çeviri
--------------------------
Ajan tahmin yapmaz, karar vermez, hesaplamaz. Elinde yalnızca dört tool var
ve hepsi LightGBM + SHAP'a bağlı. Sistem promptu ona "tool çağırmadan sayı
söyleme" der; ``faithfulness.py`` ise söylediklerini denetler.

Neden Agent Framework?
----------------------
Tool-calling döngüsünü elle yazmak mümkün (istek gönder → tool_calls'ı
ayrıştır → fonksiyonu koş → sonucu geri gönder → tekrar). Ama bu döngü
sıkıcı ve hataya açık ayrıntılarla dolu: paralel tool çağrıları, çağrı
kimlikleri, hata durumunda modele ne döneceği, oturum geçmişinin
biriktirilmesi. Microsoft Agent Framework bunların hepsini kapsıyor ve
``tools=[python_fonksiyonu]`` demek yeterli — JSON şemasını fonksiyonun tip
imzalarından ve docstring'inden kendisi üretiyor.

Senkron sarmalayıcı
-------------------
Agent Framework asenkron (``await agent.run(...)``). Streamlit ise senkron
bir yürütme modeline sahip. Bu yüzden :meth:`CreditAgent.ask` senkron bir
arayüz sunar ve asenkron çağrıyı kendi olay döngüsünde koşar.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from .config import LLMSettings, get_llm_settings
from .explainer import CreditExplainer
from .llm import LLMBackendError, check_backend_available, create_chat_client
from .prompts import SYSTEM_PROMPT
from .tools import AgentToolbox


# --------------------------------------------------------------------------
# Asenkron kodu senkron bağlamdan koşmak için yardımcı
# --------------------------------------------------------------------------
class _BackgroundLoop:
    """Uygulama ömrü boyunca yaşayan tek bir asyncio olay döngüsü.

    Neden bu gerekli?
    -----------------
    İlk denemede her soru için ``asyncio.run()`` çağırdık ve ikinci soruda
    ``RuntimeError: Event loop is closed`` aldık. Sebep şu: ``OllamaChatClient``
    içinde bir ``httpx.AsyncClient`` tutuyor ve bu istemci **oluşturulduğu
    olay döngüsüne bağlı**. ``asyncio.run()`` her çağrıda döngüyü kapattığı
    için ikinci soruda istemcinin altındaki bağlantı havuzu ölü bir döngüye
    işaret ediyordu.

    Çözüm: arka planda ``run_forever`` ile dönen tek bir döngü açmak ve tüm
    coroutine'leri ``run_coroutine_threadsafe`` ile ona göndermek. Böylece
    istemci ömrü boyunca aynı döngüde kalıyor. Bu yaklaşım Streamlit'in
    senkron yürütme modeliyle de sorunsuz çalışıyor.
    """

    _instance: _BackgroundLoop | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="xai-agent-loop", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Any, timeout: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    @classmethod
    def instance(cls) -> _BackgroundLoop:
        with cls._lock:
            if cls._instance is None or not cls._instance._thread.is_alive():
                cls._instance = cls()
            return cls._instance


def run_async(coro: Any, timeout: float | None = None) -> Any:
    """Bir coroutine'i kalıcı arka plan döngüsünde koşar ve sonucunu döndürür."""
    return _BackgroundLoop.instance().submit(coro, timeout=timeout)


@dataclass
class AgentTurn:
    """Bir soru-cevap turunun tam kaydı (denetim ve test için)."""

    question: str
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def tool_names(self) -> list[str]:
        return [c["tool"] for c in self.tool_calls]


@dataclass
class VerifiedAnswer:
    """Denetimden geçmiş (ve gerekiyorsa onarılmış) bir yanıt."""

    question: str
    turn: AgentTurn
    audit: Any
    #: İlk denemenin denetimi — onarımın etkisini ölçmek için saklanır.
    first_audit: Any = None
    repair_rounds: int = 0
    first_attempt_violations: int = 0
    attempts: int = 1

    @property
    def answer(self) -> str:
        return self.turn.answer

    @property
    def passed(self) -> bool:
        return bool(self.audit.passed)

    @property
    def was_repaired(self) -> bool:
        return self.repair_rounds > 0

    @property
    def improved(self) -> bool:
        """Onarım döngüsü ihlal sayısını gerçekten azalttı mı?"""
        return self.audit.violations < self.first_attempt_violations

    def summary(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "passed": self.passed,
            "violations": self.audit.violations,
            "first_attempt_violations": self.first_attempt_violations,
            "repair_rounds": self.repair_rounds,
            "improved": self.improved,
            "tools": self.turn.tool_names,
        }


class CreditAgent:
    """Kredi kararlarını açıklayan ajan.

    Örnek::

        from xai_agent.agent import CreditAgent
        from xai_agent.data import prepare

        ds = prepare()
        applicant = ds.X_test.iloc[0].to_dict()

        agent = CreditAgent()
        agent.set_applicant(applicant, "A-001")
        print(agent.ask("Bu başvuru neden bu kararı aldı?"))
        print(agent.ask("Vade 12 aya düşse ne olurdu?"))
    """

    def __init__(
        self,
        explainer: CreditExplainer | None = None,
        settings: LLMSettings | None = None,
        instructions: str | None = None,
    ):
        self.settings = settings or get_llm_settings()
        self.explainer = explainer or CreditExplainer()
        self.instructions = instructions or SYSTEM_PROMPT

        self._toolbox: AgentToolbox | None = None
        self._agent: Any = None
        self._session: Any = None
        self.history: list[AgentTurn] = []

    # ------------------------------------------------------------------
    # Kurulum
    # ------------------------------------------------------------------
    def health_check(self) -> tuple[bool, str]:
        """LLM arka ucunun hazır olup olmadığını bildirir."""
        return check_backend_available(self.settings)

    def set_applicant(
        self, applicant: dict[str, Any], applicant_id: str = "başvuru"
    ) -> None:
        """Ajanı belirli bir başvuruya bağlar ve sohbet geçmişini sıfırlar.

        Başvuru değiştiğinde geçmişi **mutlaka** sıfırlıyoruz. Aksi hâlde
        model önceki başvurunun sayılarını yenisine taşıyabilir — sessiz ve
        tespit edilmesi zor bir halüsinasyon kaynağı.
        """
        if self._toolbox is None:
            self._toolbox = AgentToolbox(
                explainer=self.explainer,
                applicant=dict(applicant),
                applicant_id=applicant_id,
            )
        else:
            self._toolbox.reset(applicant, applicant_id)

        self._agent = None
        self._session = None
        self.history.clear()

    @property
    def toolbox(self) -> AgentToolbox:
        if self._toolbox is None:
            raise RuntimeError(
                "Önce set_applicant(...) ile bir başvuru bağlamanız gerekiyor."
            )
        return self._toolbox

    @property
    def explanation(self):
        """Mevcut başvurunun SHAP açıklaması (LLM'e gerek yok)."""
        return self.toolbox.explanation

    def _ensure_agent(self) -> Any:
        """Agent nesnesini tembel biçimde oluşturur."""
        if self._agent is not None:
            return self._agent

        from agent_framework import Agent, AgentSession, ChatOptions

        client = create_chat_client(self.settings)
        self._agent = Agent(
            client=client,
            name="XAI-Kredi-Aciklayici",
            description=(
                "LightGBM kredi risk modelinin kararlarını SHAP çıktısına "
                "dayanarak Türkçe açıklayan asistan."
            ),
            instructions=self.instructions,
            tools=self.toolbox.build_tools(),
            default_options=ChatOptions(
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ),
        )
        self._session = AgentSession()
        return self._agent

    # ------------------------------------------------------------------
    # Sorgulama
    # ------------------------------------------------------------------
    async def ask_async(self, question: str) -> AgentTurn:
        """Ajana bir soru sorar (asenkron)."""
        agent = self._ensure_agent()
        before = len(self.toolbox.call_log)

        response = await agent.run(question, session=self._session)
        text = (getattr(response, "text", None) or "").strip()

        turn = AgentTurn(
            question=question,
            answer=text,
            tool_calls=list(self.toolbox.call_log[before:]),
        )
        self.history.append(turn)
        return turn

    def ask(self, question: str) -> str:
        """Ajana bir soru sorar ve yanıt metnini döndürür (senkron)."""
        return self.ask_turn(question).answer

    def ask_turn(self, question: str) -> AgentTurn:
        """Ajana bir soru sorar ve tur kaydının tamamını döndürür (senkron)."""
        try:
            return run_async(self.ask_async(question), timeout=self.settings.llm_timeout)
        except LLMBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - kullanıcıya anlamlı mesaj gerekiyor
            ok, msg = self.health_check()
            if not ok:
                raise LLMBackendError(
                    f"LLM arka ucuna erişilemedi.\n{msg}\nAsıl hata: {exc}"
                ) from exc
            raise

    def explain_decision(self) -> AgentTurn:
        """Standart açıklama sorusunu sorar — arayüzün ilk mesajı için."""
        return self.ask_turn(
            "Bu başvurunun sonucunu ve nedenlerini başvuru sahibine açıkla."
        )

    # ------------------------------------------------------------------
    # Denetimli sorgulama (critic-and-revise)
    # ------------------------------------------------------------------
    def audit_turn(self, turn: AgentTurn) -> Any:
        """Bir turu SHAP gerçeğine karşı denetler."""
        from .faithfulness import audit_narrative
        from .tools import what_ifs_from_calls

        return audit_narrative(
            answer=turn.answer,
            explanation=self.explanation,
            question=turn.question,
            used_tools=turn.tool_names,
            what_if_results=what_ifs_from_calls(
                self.explainer, self.toolbox.applicant, turn.tool_calls
            ),
        )

    def ask_verified(
        self, question: str, max_repairs: int = 1
    ) -> VerifiedAnswer:
        """Soruyu sorar, yanıtı denetler, ihlal varsa düzeltme ister.

        Ölçüm şunu gösterdi: prompt mühendisliği 7B'lik bir modelde tek başına
        yetmiyor (bkz. README, "Ajan sadakat ölçümü"). Ama ihlalleri
        programatik olarak *tespit* edebiliyoruz — o hâlde modele tam olarak
        neyi yanlış yaptığını söyleyip yeniden yazdırabiliriz.

        Bu, denetçiyi pasif bir ölçüm aracından **aktif bir güvenlik
        mekanizmasına** çevirir. Döngü en fazla ``max_repairs`` kez döner ve
        her zaman **en az ihlalli** denemeyi döndürür — düzeltme denemesi
        durumu kötüleştirirse ilk yanıta geri dönülür.
        """
        from .prompts import build_repair_prompt

        turn = self.ask_turn(question)
        audit = self.audit_turn(turn)
        attempts: list[tuple[AgentTurn, Any]] = [(turn, audit)]

        repairs = 0
        while repairs < max_repairs and not audit.passed:
            feedback = build_repair_prompt(audit)
            if not feedback:
                break
            repairs += 1
            turn = self.ask_turn(feedback)
            # Denetim, düzeltme mesajını değil ORİJİNAL soruyu referans alır;
            # aksi hâlde "varsayımsal soruda tool çağrısı" kuralı yanlış çalışır.
            turn.question = question
            audit = self.audit_turn(turn)
            attempts.append((turn, audit))

        best_turn, best_audit = min(attempts, key=lambda pair: pair[1].violations)
        return VerifiedAnswer(
            question=question,
            turn=best_turn,
            audit=best_audit,
            first_audit=attempts[0][1],
            repair_rounds=repairs,
            first_attempt_violations=attempts[0][1].violations,
            attempts=len(attempts),
        )

    # ------------------------------------------------------------------
    def transcript(self) -> str:
        """Sohbetin insan-okunur dökümü (rapora eklemek için)."""
        lines: list[str] = []
        for i, turn in enumerate(self.history, 1):
            lines.append(f"[{i}] SORU: {turn.question}")
            if turn.tool_calls:
                for c in turn.tool_calls:
                    status = "ok" if c["ok"] else f"HATA: {c['note']}"
                    lines.append(f"    -> tool {c['tool']}({c['args']}) [{status}]")
            else:
                lines.append("    -> (tool çağrısı YOK)")
            lines.append(f"    CEVAP: {turn.answer}")
            lines.append("")
        return "\n".join(lines)
