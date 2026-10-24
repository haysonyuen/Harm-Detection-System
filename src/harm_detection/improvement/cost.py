"""Local GPT analysis cost controls (PRD §12.7).

These are application-level estimates, not provider billing caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harm_detection.schemas.config import PricingConfig

CHARS_PER_TOKEN_ESTIMATE = 3.5  # conservative (over-estimates tokens for English text)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def cost_usd(pricing: PricingConfig, model: str, input_tokens: int, output_tokens: int) -> float:
    price = pricing.models.get(model)
    if price is None:
        raise KeyError(f"no pricing configured for model {model!r}; add it to the pricing config")
    return (
        input_tokens * price.input_per_million_usd + output_tokens * price.output_per_million_usd
    ) / 1e6


@dataclass(frozen=True)
class BudgetCheck:
    allowed: bool
    budget_usd: float
    spent_usd: float
    estimated_call_usd: float
    remaining_usd: float
    reason: str | None = None


def check_budget(
    *,
    pricing: PricingConfig,
    model: str,
    prompt_text: str,
    reserved_output_tokens: int,
    spent_usd: float,
    budget_usd: float,
    calls: int = 1,
) -> BudgetCheck:
    est = calls * cost_usd(pricing, model, estimate_tokens(prompt_text), reserved_output_tokens)
    remaining = budget_usd - spent_usd
    if spent_usd + est > budget_usd:
        return BudgetCheck(
            False,
            budget_usd,
            spent_usd,
            est,
            remaining,
            f"estimated cost ${est:.4f} would exceed the remaining local budget ${remaining:.4f}",
        )
    return BudgetCheck(True, budget_usd, spent_usd, est, remaining)
