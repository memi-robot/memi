"""The list prices of every model the released runs called, and the dollars of one call."""
from __future__ import annotations

#: $ per million tokens of fresh input, cached input, cache writes and output.
#: GPT-6 Astra as in the base harness, the others at their list prices.
PRICES = {"gpt-6-astra": (10.0, 1.0, 12.5, 50.0), "gpt-6-luna": (0.10, 0.01, 0.125, 0.50),
          "claude-fable-5-1": (10.0, 0.25, 12.5, 50.0), "claude-haiku-5-5": (0.10, 0.01, 0.125, 0.50)}


def dollars(model: str, input_tokens: int, cached: int, written: int, output: int) -> float:
    """$ of one call. `input_tokens` counts every input token, the cached and the written ones included."""
    fresh, cached_price, write_price, output_price = PRICES[model]
    return ((input_tokens - cached - written) * fresh + cached * cached_price + written * write_price
            + output * output_price) / 1e6
