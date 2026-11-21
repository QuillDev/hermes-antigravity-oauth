"""Exact native model IDs and optional CLI effort flags for Antigravity."""

from __future__ import annotations

from typing import Any

# Compatibility exports used by client.py; neither invents account models.
_FALLBACK_MODELS: tuple[str, ...] = ()
_MODEL_ALIASES: dict[str, str] = {}

# Official agy --effort vocabulary, not a model availability table.
_CLI_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def model_effort(model: str | None) -> str | None:
    """Recognize a native effort suffix without changing the inference ID."""
    suffix = (model or "").rpartition("-")[2]
    return suffix if suffix in _CLI_EFFORTS else None


def supports_effort_flag(model: str | None) -> bool:
    """Only manual suffixless Gemini requests accept the separate CLI flag."""
    return bool(model and model.startswith("gemini-") and not model_effort(model))


def _normalize_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    value = str(effort).strip().lower()
    if value in ("none", "off", "minimal"):
        return "low"
    return value if value in _CLI_EFFORTS else None


def resolve_model_and_effort(
    model: str | None,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None]:
    """Pass selected IDs verbatim; native variants own their effort.

    Manual suffixless Gemini requests may use agy's --effort vocabulary.
    No generated IDs, aliases, inherited config or implicit model/effort defaults.
    An omitted model lets the official CLI choose its own default.
    """
    selected = model or ""
    effort = _normalize_effort(reasoning_effort) if supports_effort_flag(selected) else None
    return selected, effort


class AntigravityModelCapabilities(dict):
    """Dynamic per-model capability map for ``ProviderProfile.model_capabilities``.

    ``agy`` serves rotating native model IDs (e.g. ``gemini-4-neon-high``,
    ``claude-sonnet-5-5-high``, ``gpt-oss-120b-medium``) that are not indexed
    in models.dev. Core looks up ``profile.model_capabilities.get(model, {})``
    by exact ID; subclassing ``dict`` resolves current and future Gemini/Claude
    models as vision-capable (1M window) while keeping GPT-OSS text-only.
    """

    @staticmethod
    def _classify(model: Any) -> dict[str, Any] | None:
        if not isinstance(model, str):
            return None
        key = model.strip().lower()
        if not key:
            return None
        if key.startswith("gpt-oss"):
            return {
                "supports_vision": False,
                "supports_tools": True,
                "supports_reasoning": True,
                "context_window": 131_072,
            }
        if key.startswith(("gemini-", "claude-")):
            return {
                "supports_vision": True,
                "supports_tools": True,
                "supports_reasoning": True,
                "context_window": 1_000_000,
            }
        return None

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        classified = self._classify(key)
        return classified if classified is not None else default

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or self._classify(key) is not None

    def __getitem__(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        classified = self._classify(key)
        if classified is not None:
            return classified
        raise KeyError(key)


MODEL_CAPABILITIES: dict[str, dict[str, Any]] = AntigravityModelCapabilities()
