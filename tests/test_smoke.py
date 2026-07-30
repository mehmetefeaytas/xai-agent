"""Duman testi (smoke test) — kurulum ayakta mı?

"Smoke test" terimi elektronikten gelir: devreye ilk kez akım verip duman
çıkıp çıkmadığına bakmak. Amaç doğruluk değil, *çalışıyor mu* kontrolü.

Bu testlerin varlığı, Faz 1'de bir hata aldığımızda "kurulum mu bozuk,
kodum mu?" sorusunu ikiye ayırır.
"""

from __future__ import annotations

import sys

import pytest

REQUIRED = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "shap",
    "numba",
    "pydantic",
    "streamlit",
    "matplotlib",
    "joblib",
]


def test_python_version() -> None:
    """Python 3.12 serisinde olmalıyız (LightGBM/SHAP wheel desteği için)."""
    assert sys.version_info[:2] == (3, 12), (
        f"Beklenen Python 3.12, bulunan {sys.version_info[:2]}. "
        "uv python pin 3.12 komutunu çalıştırın."
    )


@pytest.mark.parametrize("module_name", REQUIRED)
def test_dependency_importable(module_name: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    assert getattr(module, "__version__", None), f"{module_name} sürümü okunamadı"


def test_lightgbm_openmp_works() -> None:
    """LightGBM gerçekten eğitim yapabiliyor mu?

    macOS'ta ``libomp`` kurulu değilse LightGBM import edilir ama eğitimde
    ``Library not loaded: @rpath/libomp.dylib`` hatası verir. Bu test o
    durumu import aşamasında değil, gerçek eğitimde yakalar.
    """
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=80) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=5, num_leaves=3, verbosity=-1)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (80,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_package_importable() -> None:
    import xai_agent

    assert xai_agent.__version__


def test_agent_framework_importable() -> None:
    """Microsoft Agent Framework ve iki yerel arka uç kurulu mu?"""
    from agent_framework import Agent, AgentSession, ChatOptions  # noqa: F401
    from agent_framework.ollama import OllamaChatClient  # noqa: F401

    assert Agent is not None


def test_foundry_local_importable() -> None:
    """Faz 5 geçişi için Foundry Local istemcisi kurulu mu?"""
    from agent_framework.foundry import FoundryLocalClient  # noqa: F401

    assert FoundryLocalClient is not None


@pytest.mark.llm
def test_ollama_reachable(ollama_available: bool) -> None:
    """Ollama ayakta ve yapılandırılmış model mevcut mu?

    Faz 0-2'yi kırmaması için isteğe bağlı: Ollama kapalıysa atlanır.
    """
    if not ollama_available:
        pytest.skip("Ollama çalışmıyor veya model yok (ollama serve ile başlatın)")
    from xai_agent.llm import check_backend_available

    ok, msg = check_backend_available()
    assert ok, msg
