"""LLM istemci fabrikası: Ollama / Foundry Local / Azure OpenAI arası geçiş.

Neden soyutlama katmanı?
------------------------
Projenin hedefi Microsoft yığınında (Azure AI Foundry Local) çalışmak.
Ancak Faz 1-2'de LLM'e hiç ihtiyaç yoktu ve Faz 3'te geliştirmeye Ollama ile
başladık — çünkü model zaten kuruluydu ve iterasyon hızı önemliydi.

Bu dosya sayesinde arka uç değişikliği **tek satırlık bir ortam değişkeni**:

    XAI_LLM_BACKEND=foundry   # veya ollama / azure

Ajan kodu hangi arka ucun konuştuğunu bilmez. Bu, Faz 5'teki Foundry Local
geçişini kod değişikliği olmadan yapılabilir kılıyor.

Üç arka uç
----------
``ollama``
    Yerel, ``OllamaChatClient``. Varsayılan. Model: ``qwen2.5:7b-instruct``.
``foundry``
    Azure AI Foundry Local, ``FoundryLocalClient``. Yerel çalışır, Microsoft
    yığınının parçasıdır, ONNX Runtime ile donanım hızlandırma kullanır.
``azure``
    Azure OpenAI (bulut), ``OpenAIChatClient`` üzerinden OpenAI-uyumlu
    endpoint. API anahtarı gerektirir.
"""

from __future__ import annotations

from typing import Any

from .config import LLMSettings, get_llm_settings

SUPPORTED_BACKENDS = ("ollama", "foundry", "azure")


class LLMBackendError(RuntimeError):
    """LLM arka ucu kullanılamadığında atılır — mesajı kullanıcıya gösterilir."""


def describe_backend(settings: LLMSettings | None = None) -> dict[str, str]:
    """Aktif arka ucun insan-okunur özeti (arayüzde gösterilir)."""
    s = settings or get_llm_settings()
    return {
        "backend": s.llm_backend,
        "model": s.llm_model,
        "base_url": s.llm_base_url if s.llm_backend != "foundry" else "(yerel Foundry)",
        "temperature": str(s.llm_temperature),
    }


def create_chat_client(settings: LLMSettings | None = None) -> Any:
    """Yapılandırmaya göre bir Agent Framework sohbet istemcisi üretir."""
    s = settings or get_llm_settings()
    backend = s.llm_backend.strip().lower()

    if backend not in SUPPORTED_BACKENDS:
        raise LLMBackendError(
            f"Bilinmeyen arka uç: {s.llm_backend!r}. "
            f"Desteklenenler: {', '.join(SUPPORTED_BACKENDS)}"
        )

    if backend == "ollama":
        try:
            from agent_framework.ollama import OllamaChatClient
        except ImportError as exc:  # pragma: no cover
            raise LLMBackendError(
                "agent-framework-ollama kurulu değil:  uv add agent-framework-ollama"
            ) from exc
        host = s.llm_base_url.removesuffix("/v1").rstrip("/")
        return OllamaChatClient(host=host, model=s.llm_model)

    if backend == "foundry":
        try:
            from agent_framework.foundry import FoundryLocalClient
        except ImportError as exc:  # pragma: no cover
            raise LLMBackendError(
                "agent-framework-foundry-local kurulu değil:  "
                "uv add agent-framework-foundry-local"
            ) from exc
        # Foundry Local modeli kendisi indirir/başlatır (bootstrap=True).
        return FoundryLocalClient(model=s.llm_model)

    try:
        from agent_framework.openai import OpenAIChatClient
    except ImportError as exc:  # pragma: no cover
        raise LLMBackendError(
            "agent-framework-openai kurulu değil:  uv add agent-framework-openai"
        ) from exc
    return OpenAIChatClient(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model_id=s.llm_model,
    )


def check_backend_available(settings: LLMSettings | None = None) -> tuple[bool, str]:
    """Arka ucun ayakta olup olmadığını *ağ çağrısıyla* sınar.

    Arayüzün "LLM kapalı" durumunu kullanıcıya net göstermesi için var.
    Sessizce başarısız olup boş yanıt döndürmek en kötü davranış olurdu.

    Returns:
        ``(kullanilabilir_mi, mesaj)``
    """
    s = settings or get_llm_settings()
    backend = s.llm_backend.strip().lower()

    if backend == "ollama":
        import json
        import urllib.error
        import urllib.request

        host = s.llm_base_url.removesuffix("/v1").rstrip("/")
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, (
                f"Ollama'ya {host} adresinde ulaşılamıyor ({exc}). "
                "Çalıştırmak için:  ollama serve"
            )
        available = [m.get("name", "") for m in tags.get("models", [])]
        if s.llm_model not in available:
            return False, (
                f"'{s.llm_model}' modeli Ollama'da yok. Mevcut modeller: "
                f"{', '.join(available) or '(yok)'}. "
                f"İndirmek için:  ollama pull {s.llm_model}"
            )
        return True, f"Ollama hazır — model: {s.llm_model}"

    if backend == "foundry":
        try:
            from foundry_local import FoundryLocalManager  # noqa: F401
        except ImportError:
            return False, (
                "foundry-local-sdk kurulu değil:  uv add agent-framework-foundry-local"
            )
        import shutil

        if shutil.which("foundry") is None:
            return False, (
                "Foundry Local CLI bulunamadı. macOS kurulumu:  "
                "brew tap microsoft/foundrylocal && brew install foundrylocal"
            )
        return True, f"Foundry Local CLI bulundu — model: {s.llm_model}"

    if not s.llm_api_key or s.llm_api_key == "not-needed-for-local":
        return False, (
            "Azure OpenAI için API anahtarı gerekli. .env dosyasına ekleyin:  "
            "XAI_LLM_API_KEY=..."
        )
    return True, f"Azure OpenAI yapılandırıldı — model: {s.llm_model}"
