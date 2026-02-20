"""Pure-function span extraction from document text."""

from __future__ import annotations

import re


def _snap_to_sentence_start(text: str, pos: int) -> int:
    """Move *pos* forward to the start of the next sentence if it's mid-sentence.

    Looks for '. ', '! ', '? ' (or start-of-string) boundaries.
    If no boundary is found, returns *pos* unchanged.
    """
    if pos == 0:
        return 0
    # Search forward from pos for sentence-start patterns
    m = re.search(r"(?<=[.!?])\s+", text[pos:])
    if m and m.start() < 60:
        return pos + m.end()
    return pos


def extract_span(
    document_text: str,
    match_start: int,
    match_end: int,
    span_length: int = 256,
) -> tuple[str, str]:
    """Extract a (prompt, completion) pair from *document_text*.

    Parameters
    ----------
    document_text:
        Full text of the source document.
    match_start:
        Character offset where the matched completion begins.
    match_end:
        Character offset where the matched completion ends.
    span_length:
        Desired prompt length in characters (before the match).

    Returns
    -------
    (prompt, completion) – The prompt is the text preceding the match,
    snapped to sentence boundaries when possible.  The completion is
    ``document_text[match_start:match_end]``.
    """
    completion = document_text[match_start:match_end]
    # Ensure leading space for correct SentencePiece tokenization.
    if completion and not completion.startswith(" "):
        completion = " " + completion

    raw_start = max(0, match_start - span_length)
    # Snap to a sentence boundary (move forward to avoid partial sentences)
    if raw_start > 0:
        snapped = _snap_to_sentence_start(document_text, raw_start)
        # Only snap if we still have meaningful prompt left
        if snapped < match_start:
            raw_start = snapped

    prompt = document_text[raw_start:match_start]
    return prompt, completion
