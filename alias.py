"""Swap the word "hermes" for an alias on the way to agy, and back on the way out.

Google's backend answers some prompts that name Hermes (its persona line, docs URL and steering
section) with a fake ``RESOURCE_EXHAUSTED (429)`` after minutes of waiting (issue #27). The same
text with the word swapped goes through, so every prompt and rules file is masked before agy sees
it, and every reply, reasoning delta and tool call is unmasked before Hermes sees it.
"""

from __future__ import annotations

import re

ALIAS = "personal-nako-agent"
_WORD = re.compile("hermes", re.IGNORECASE)
_ALIAS = re.compile(re.escape(ALIAS), re.IGNORECASE)


def _same_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return "-".join(part.capitalize() for part in target.split("-"))
    return target


def mask(text: str) -> str:
    return _WORD.sub(lambda m: _same_case(m.group(0), ALIAS), text)


def unmask(text: str) -> str:
    return _ALIAS.sub(lambda m: _same_case(m.group(0), "hermes"), text)


class StreamUnmasker:
    """Unmask streamed deltas, holding back a tail that may be the start of a split alias."""

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        buf = self._held + text
        lowered = buf.lower()
        keep = next((k for k in range(min(len(ALIAS) - 1, len(buf)), 0, -1)
                     if lowered.endswith(ALIAS[:k])), 0)
        self._held = buf[len(buf) - keep:] if keep else ""
        return unmask(buf[:len(buf) - keep])

    def flush(self) -> str:
        held, self._held = self._held, ""
        return held
