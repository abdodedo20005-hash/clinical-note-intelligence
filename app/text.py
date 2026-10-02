import re


def clean_note_text(text: str) -> str:
    """Same preprocessing as the notebook: replace non-ASCII characters and collapse whitespace."""
    text = re.sub(r"[^\x20-\x7E\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()
