"""Unit tests for span extraction."""

from __future__ import annotations

from backend.span import extract_span


def test_basic_extraction():
    text = "The quick brown fox jumps over the lazy dog."
    prompt, completion = extract_span(text, 16, 25, span_length=16)
    assert completion == "fox jumps"
    assert prompt == text[:16]


def test_match_at_start():
    text = "Hello world. This is a test."
    prompt, completion = extract_span(text, 0, 5, span_length=100)
    assert completion == "Hello"
    assert prompt == ""


def test_match_at_end():
    text = "First sentence. Second sentence. Final word."
    match_start = len(text) - 11  # "Final word."
    match_end = len(text)
    prompt, completion = extract_span(text, match_start, match_end, span_length=100)
    assert completion == "Final word."
    assert len(prompt) > 0


def test_span_larger_than_document():
    text = "Short text."
    prompt, completion = extract_span(text, 6, 11, span_length=1000)
    assert completion == "text."
    assert prompt == "Short "


def test_sentence_snapping():
    text = "First sentence. Second sentence. Third sentence. The match here."
    # match_start is at "The match here."
    match_start = text.index("The match here.")
    match_end = len(text)
    # Use a span_length that puts raw_start mid-sentence
    prompt, completion = extract_span(text, match_start, match_end, span_length=30)
    assert completion == "The match here."
    # Prompt should snap to a sentence boundary
    assert not prompt or prompt[0].isupper() or prompt.startswith(" ")


def test_zero_span_length():
    text = "Hello world."
    prompt, completion = extract_span(text, 6, 12, span_length=0)
    assert completion == "world."
    assert prompt == ""


def test_exact_match_boundaries():
    text = "ABCDEFGHIJ"
    prompt, completion = extract_span(text, 5, 10, span_length=5)
    assert completion == "FGHIJ"
    assert prompt == "ABCDE"
