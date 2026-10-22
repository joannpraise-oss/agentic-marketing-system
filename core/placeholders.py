import re
from typing import Any, Iterable

# A brand-specific fact the brief didn't give us, e.g. "[▢ return window, e.g. 30 days]".
PLACEHOLDER_RE = re.compile(r"\[▢[^\[\]]*\]")


class PlaceholdersRemainError(Exception):
    """An asset can't be approved while it still contains placeholders."""

    def __init__(self, placeholders: list[str]):
        super().__init__(f"{len(placeholders)} placeholder(s) still need input")
        self.placeholders = placeholders


def find_placeholders(texts: Iterable[str | None]) -> list[str]:
    """Unique placeholders in order of first appearance."""
    found: list[str] = []
    for text in texts:
        for match in PLACEHOLDER_RE.findall(text or ""):
            if match not in found:
                found.append(match)
    return found


def asset_texts(caption: str | None, alt_text: str | None, faq: Any) -> list[str | None]:
    """Every user-facing text field of a creative, including each FAQ question and answer."""
    texts = [caption, alt_text]
    for entry in faq or []:
        if isinstance(entry, dict):
            texts += [entry.get("question"), entry.get("answer")]
        else:
            texts += [getattr(entry, "question", None), getattr(entry, "answer", None)]
    return texts
