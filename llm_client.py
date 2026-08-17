"""
Thin wrapper around the Groq SDK.

Consolidates three things that existed as copy-pasted logic across the
original app:
  1. Retry-with-backoff on rate-limit errors (_call_with_retry)
  2. Robust JSON extraction from LLM responses (_robust_json_parse)
  3. A hard request timeout, which the original code never set — a hung
     Groq call used to hang the entire Streamlit session with no way out.

Every LLM call in the app should go through JobFitLLMClient.complete_json()
rather than instantiating groq.Groq() directly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from groq import Groq

from config import Settings, get_logger

logger = get_logger(__name__)


class LLMResponseError(Exception):
    """Raised when the LLM response can't be parsed into usable JSON."""


@dataclass
class LLMResult:
    data: Any               # parsed list or dict
    raw_text: str
    model: str


class JobFitLLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        label: str = "unlabeled",
        retry_on_rate_limit: bool = True,
    ) -> LLMResult:
        """
        Call the model and parse the response as JSON (list or dict).
        Retries on rate-limit errors with exponential backoff.
        """
        fn = self._complete_json_once
        if not retry_on_rate_limit:
            return fn(prompt, model, temperature, max_tokens, label)

        return self._with_retry(
            fn, prompt, model, temperature, max_tokens, label,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _complete_json_once(
        self, prompt: str, model: str, temperature: float, max_tokens: int, label: str
    ) -> LLMResult:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
        parsed = self._robust_json_parse(raw, label=label)
        return LLMResult(data=parsed, raw_text=raw, model=model)

    def _with_retry(self, fn, *args, max_retries: int | None = None):
        max_retries = max_retries or self._settings.llm_max_retries
        last_err: Exception | None = None

        for attempt in range(max_retries):
            try:
                return fn(*args)
            except Exception as e:  # noqa: BLE001 — intentionally broad; classified below
                err_str = str(e).lower()
                is_rate_limit = any(
                    token in err_str for token in ("rate", "429", "quota", "too many")
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 8 * (attempt + 1)
                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %ds",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    last_err = e
                    continue
                logger.error("LLM call failed (non-retryable or retries exhausted): %s", e)
                raise
        raise last_err  # pragma: no cover — unreachable, satisfies type checkers

    @staticmethod
    def _robust_json_parse(raw: str, *, label: str = "") -> Any:
        """
        Extract valid JSON (array or object) from LLM output.

        Ported from the original _robust_json_parse with the same strategy
        order (markdown fence stripping -> outermost array/object detection
        -> comment stripping -> trailing comma removal), since that ordering
        was itself a bug fix for array-returning prompts being silently
        destroyed by an object-only search.
        """
        if not raw or not raw.strip():
            raise LLMResponseError(f"[{label}] LLM returned an empty response")

        text = raw.strip()

        for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break

        arr_start, arr_end = text.find("["), text.rfind("]")
        obj_start, obj_end = text.find("{"), text.rfind("}")

        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            if obj_start == -1 or arr_start < obj_start:
                text = text[arr_start: arr_end + 1]
            elif obj_end > obj_start:
                text = text[obj_start: obj_end + 1]
        elif obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            text = text[obj_start: obj_end + 1]

        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"[{label}] JSON parse failed after cleanup: {e}. "
                f"Cleaned text (first 300 chars): {text[:300]!r}"
            ) from e
