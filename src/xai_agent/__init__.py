"""XAI-Agent — SHAP tabanlı şeffaf kredi risk açıklama ajanı.

Üç katmanlı mimari:

1. :mod:`xai_agent.model`      — LightGBM tahmin üretir
2. :mod:`xai_agent.explainer`  — SHAP tahminin "neden"ini sayısallaştırır
3. :mod:`xai_agent.agent`      — LLM ajanı sayıları doğal dile çevirir

Katmanların sorumlulukları kesin olarak ayrıdır: LLM **asla** tahmin
yapmaz, yalnızca SHAP çıktısını aktarır. :mod:`xai_agent.faithfulness`
modülü bu sözleşmenin ihlal edilip edilmediğini ölçer.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
