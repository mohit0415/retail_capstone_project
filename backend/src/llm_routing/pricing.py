"""Model price table and cost estimation.

Prices are USD per one million tokens, input and output separately. The
built-in table covers the deployments this project uses; ``MODEL_PRICES_JSON``
in ``.env`` overrides or extends it without a code change (Azure list prices
move, and a deployment name can differ from the model name).

Lookups are tolerant: ``gpt-4o-mini-2024-07-18`` resolves to ``gpt-4o-mini``
by longest-prefix match, and an unknown model prices at zero with a warning
logged once per name, so a missing row never breaks a request.
"""

import json
import logging
import threading
from dataclasses import dataclass

from configs.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-35-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "simple-agent": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "ollama/llama3.2": {"input": 0.0, "output": 0.0},
}


@dataclass(frozen=True, slots=True)
class Price:
    model: str
    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            (prompt_tokens * self.input_per_million + completion_tokens * self.output_per_million)
            / 1_000_000,
            8,
        )


_table: dict[str, dict[str, float]] | None = None
_unknown_reported: set[str] = set()
_lock = threading.Lock()


def _load_table() -> dict[str, dict[str, float]]:
    global _table

    if _table is not None:
        return _table

    with _lock:
        if _table is not None:
            return _table

        table = {key.lower(): dict(value) for key, value in DEFAULT_PRICES.items()}
        raw = (settings.model_prices_json or "").strip()

        if raw:
            try:
                override = json.loads(raw)

                for name, prices in override.items():
                    table[str(name).lower()] = {
                        "input": float(prices.get("input", 0.0)),
                        "output": float(prices.get("output", 0.0)),
                    }

                logger.info("model price table: %d override(s) loaded from MODEL_PRICES_JSON", len(override))
            except (ValueError, AttributeError, TypeError) as exc:
                logger.warning("MODEL_PRICES_JSON could not be parsed (%s); using the built-in table", exc)

        _table = table

        return table


def reset_price_table() -> None:
    """Forget the loaded table so the next lookup re-reads settings (tests)."""
    global _table

    with _lock:
        _table = None
        _unknown_reported.clear()


def price_for(model: str | None) -> Price:
    table = _load_table()
    name = (model or "").strip().lower()

    if not name:
        return Price("unknown", 0.0, 0.0)

    if name in table:
        row = table[name]

        return Price(name, row["input"], row["output"])

    stripped = name.split("/", 1)[-1]
    candidates = [key for key in table if stripped.startswith(key) or name.startswith(key)]

    if candidates:
        best = max(candidates, key=len)
        row = table[best]

        return Price(best, row["input"], row["output"])

    with _lock:
        if name not in _unknown_reported:
            _unknown_reported.add(name)
            logger.warning(
                "no price known for model %r; it is counted at $0. Add it to MODEL_PRICES_JSON", model
            )

    return Price(name, 0.0, 0.0)


def estimate_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    return price_for(model).cost(int(prompt_tokens or 0), int(completion_tokens or 0))


def tier_prices() -> dict[str, dict[str, float]]:
    """The two configured tiers priced, for the metrics report."""
    small = price_for(settings.azure_openai_small_deployment)
    strong = price_for(settings.azure_openai_strong_deployment)

    return {
        "small": {
            "deployment": settings.azure_openai_small_deployment,
            "input_per_million": small.input_per_million,
            "output_per_million": small.output_per_million,
        },
        "strong": {
            "deployment": settings.azure_openai_strong_deployment,
            "input_per_million": strong.input_per_million,
            "output_per_million": strong.output_per_million,
        },
    }
