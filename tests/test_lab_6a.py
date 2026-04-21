"""Autograder tests for Lab 6A — NER Pipeline."""

import pytest
import pandas as pd
import numpy as np
import spacy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ner_pipeline import (load_data, explore_data, preprocess_text,
                          extract_spacy_entities, extract_hf_entities,
                          compare_ner_outputs, evaluate_ner)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_entities.csv")


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def hf_ner():
    from transformers import pipeline
    return pipeline("ner", model="dslim/bert-base-NER")


@pytest.fixture
def df():
    data = load_data(DATA_PATH)
    assert data is not None, "load_data returned None"
    return data


SAMPLE_NER_TEXT = (
    "The IPCC released a report on climate change impacts in Jordan "
    "and the broader Middle East region in March 2024."
)


def test_data_loaded(df):
    """Dataset loads with expected shape and columns."""
    assert df.shape[0] > 50, f"Expected >50 rows, got {df.shape[0]}"
    required_cols = {"id", "text", "source", "language", "category"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )


def test_explore_data(df):
    """explore_data returns a summary dict with required keys and sensible values."""
    result = explore_data(df)
    assert result is not None, "explore_data returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    for key in ["shape", "lang_counts", "category_counts", "text_length_stats"]:
        assert key in result, f"Missing '{key}' key"

    # shape: (rows, cols) matching the DataFrame
    assert tuple(result["shape"]) == tuple(df.shape), (
        f"shape {result['shape']} != df.shape {df.shape}"
    )

    # language counts sum to total rows
    assert sum(result["lang_counts"].values()) == len(df), (
        "lang_counts should cover every row"
    )

    # category counts sum to total rows
    assert sum(result["category_counts"].values()) == len(df), (
        "category_counts should cover every row"
    )

    # text length stats present and ordered
    stats = result["text_length_stats"]
    for key in ["mean", "min", "max"]:
        assert key in stats, f"text_length_stats missing '{key}'"
    assert stats["min"] <= stats["mean"] <= stats["max"], (
        f"Text length stats out of order: {stats}"
    )


def test_preprocess_text_basic(nlp):
    """preprocess_text returns lowercase, no punctuation, lemmatized tokens."""
    result = preprocess_text("The IPCC released its latest report.", nlp)
    assert result is not None, "preprocess_text returned None"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list"

    for token in result:
        assert isinstance(token, str), f"Token must be str, got {type(token)}"
        assert token == token.lower(), f"Token '{token}' is not lowercase"

    for token in result:
        assert not all(c in ".,;:!?()-'\"" for c in token), (
            f"Punctuation token '{token}' should be removed"
        )


def test_preprocess_handles_unicode(nlp):
    """preprocess_text handles non-ASCII characters without crashing."""
    result = preprocess_text("Café résumé naïve Zürich — 2°C target", nlp)
    assert result is not None, "preprocess_text returned None on Unicode input"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list on Unicode input"


def test_spacy_ner_returns_dataframe(nlp):
    """extract_spacy_entities returns a DataFrame with required columns."""
    sample_df = pd.DataFrame({
        "id": [1],
        "text": [SAMPLE_NER_TEXT],
        "source": ["test"],
        "language": ["en"],
        "category": ["policy"],
    })
    result = extract_spacy_entities(sample_df, nlp)
    assert result is not None, "extract_spacy_entities returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label", "start_char", "end_char"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )


def test_spacy_ner_finds_entities(nlp):
    """spaCy NER finds at least one known entity and filters non-English rows."""
    sample_df = pd.DataFrame({
        "id": [1, 2],
        "text": [SAMPLE_NER_TEXT, "نص عربي لا ينبغي أن يُعالج"],
        "source": ["test", "test"],
        "language": ["en", "ar"],
        "category": ["policy", "policy"],
    })
    result = extract_spacy_entities(sample_df, nlp)
    assert result is not None, "extract_spacy_entities returned None"
    assert len(result) > 0, "No entities found in text containing IPCC, Jordan, etc."
    # Only the English row (id=1) should have produced entities
    assert set(result["text_id"].unique()).issubset({1}), (
        f"Non-English rows should be filtered. Got text_ids: {set(result['text_id'].unique())}"
    )

    entity_texts = result["entity_text"].str.lower().tolist()
    assert any(term in " ".join(entity_texts) for term in
               ["ipcc", "jordan", "middle east", "march"]), (
        f"Expected known entities. Found: {result['entity_text'].tolist()}"
    )


def test_hf_ner_returns_dataframe(hf_ner):
    """extract_hf_entities returns a DataFrame with required columns and clean labels."""
    sample_df = pd.DataFrame({
        "id": [1],
        "text": [SAMPLE_NER_TEXT],
        "source": ["test"],
        "language": ["en"],
        "category": ["policy"],
    })
    result = extract_hf_entities(sample_df, hf_ner)
    assert result is not None, "extract_hf_entities returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label", "start_char", "end_char"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )

    if len(result) > 0:
        # Labels must be stripped of IOB prefixes (B-/I-)
        for label in result["entity_label"]:
            assert not label.startswith("B-"), (
                f"IOB prefix 'B-' not stripped from label '{label}'"
            )
            assert not label.startswith("I-"), (
                f"IOB prefix 'I-' not stripped from label '{label}'"
            )


def test_entity_comparison():
    """compare_ner_outputs produces counts, totals, and overlap sets."""
    spacy_df = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "March 2024"],
        "entity_label": ["ORG", "GPE", "DATE"],
        "start_char": [4, 50, 80],
        "end_char": [8, 56, 90],
    })
    hf_df = pd.DataFrame({
        "text_id": [1, 1],
        "entity_text": ["IPCC", "Jordan"],
        "entity_label": ["ORG", "LOC"],
        "start_char": [4, 50],
        "end_char": [8, 56],
    })
    result = compare_ner_outputs(spacy_df, hf_df)
    assert result is not None, "compare_ner_outputs returned None"
    assert isinstance(result, dict), "Must return a dictionary"

    for key in ["spacy_counts", "hf_counts", "total_spacy", "total_hf",
                "both", "spacy_only", "hf_only"]:
        assert key in result, f"Missing '{key}' key"

    assert result["total_spacy"] == 3, f"Expected 3 spaCy entities, got {result['total_spacy']}"
    assert result["total_hf"] == 2, f"Expected 2 HF entities, got {result['total_hf']}"

    # Overlap: (1, 'IPCC') and (1, 'Jordan') found by both; (1, 'March 2024') only spaCy
    both = {tuple(x) for x in result["both"]}
    spacy_only = {tuple(x) for x in result["spacy_only"]}
    hf_only = {tuple(x) for x in result["hf_only"]}

    assert both == {(1, "IPCC"), (1, "Jordan")}, (
        f"both should match on (text_id, entity_text). Got: {both}"
    )
    assert spacy_only == {(1, "March 2024")}, (
        f"spacy_only should contain the DATE entity. Got: {spacy_only}"
    )
    assert hf_only == set(), f"hf_only should be empty for this fixture. Got: {hf_only}"


def test_evaluate_ner_metrics():
    """evaluate_ner computes precision, recall, F1 in 0-1 range."""
    predicted = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "March 2024"],
        "entity_label": ["ORG", "GPE", "DATE"],
    })
    gold = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "Middle East"],
        "entity_label": ["ORG", "GPE", "LOC"],
    })
    result = evaluate_ner(predicted, gold)
    assert result is not None, "evaluate_ner returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    for key in ["precision", "recall", "f1"]:
        assert key in result, f"Missing '{key}' key"
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} not in [0, 1]"
    assert result["precision"] > 0, "Precision should be > 0 with matching entities"
    assert result["recall"] > 0, "Recall should be > 0 with matching entities"
