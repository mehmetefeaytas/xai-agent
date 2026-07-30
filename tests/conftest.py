"""Ortak test düzeneği (fixtures).

Model ve veri yüklemesi pahalı olduğu için ``session`` kapsamında
önbelleklenir — tüm test dosyaları aynı nesneleri paylaşır.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def dataset():
    from xai_agent.data import prepare

    return prepare()


@pytest.fixture(scope="session")
def bundle():
    from xai_agent.model import ModelBundle

    try:
        return ModelBundle.load()
    except FileNotFoundError:
        pytest.skip(
            "Eğitilmiş model yok. Önce:  uv run python scripts/train.py"
        )


@pytest.fixture(scope="session")
def explainer(bundle):
    from xai_agent.explainer import CreditExplainer

    return CreditExplainer(bundle)


@pytest.fixture(scope="session")
def sample_applicant(dataset):
    """Test setinden tek bir başvuru (sözlük olarak)."""
    return dataset.X_test.iloc[0].to_dict()


@pytest.fixture(scope="session")
def sample_explanation(explainer, dataset):
    return explainer.explain_frame(dataset.X_test.iloc[[0]], applicant_id="T-000")


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    from xai_agent.llm import check_backend_available

    ok, _ = check_backend_available()
    return ok


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: LLM arka ucu gerektiren testler (yavaş, ağ/yerel servis)"
    )
