"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    return pd.read_csv(filepath)


def preprocess_text(text):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using spaCy.

    Args:
        text: Raw text string.

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: Process text with spaCy, filter punctuation and whitespace,
    #       return lowercased lemmas
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]


def extract_spacy_entities(texts):
    """Extract named entities from texts using spaCy NER.

    Args:
        texts: List of (text_id, text_string) tuples.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Process each text with spaCy, collect entities into rows,
    #       return as a DataFrame
    nlp = spacy.load("en_core_web_sm")
    rows = []
    for text_id, text_str in texts:
        doc = nlp(text_str)
        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
    return pd.DataFrame(rows)


def extract_hf_entities(texts):
    """Extract named entities from texts using Hugging Face NER.

    Uses the dslim/bert-base-NER model.

    Args:
        texts: List of (text_id, text_string) tuples.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Create an HF NER pipeline with dslim/bert-base-NER,
    #       process each text, reconstruct entity spans from subword
    #       tokens, return as a DataFrame
    ner_pipeline = hf_pipeline("ner", model="dslim/bert-base-NER")
    rows = []
    for text_id, text_str in texts:
        entities = ner_pipeline(text_str)
        
        # Reconstruct full entity text from subword tokens
        current_entity = None
        for token_info in entities:
            token = token_info["word"]
            label = token_info["entity"]
            score = token_info["score"]
            start_idx = token_info["start"]
            end_idx = token_info["end"]
            
            # Handle B- (beginning) and I- (inside) tags
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    rows.append({
                        "text_id": text_id,
                        "entity_text": current_entity["text"],
                        "entity_label": current_entity["label"],
                        "start_char": current_entity["start"],
                        "end_char": current_entity["end"]
                    })
                # Start new entity
                current_entity = {
                    "text": token.replace("##", ""),
                    "label": label[2:],  # Remove B- prefix
                    "start": start_idx,
                    "end": end_idx
                }
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += token.replace("##", "")
                current_entity["end"] = end_idx
        
        # Save last entity if exists
        if current_entity:
            rows.append({
                "text_id": text_id,
                "entity_text": current_entity["text"],
                "entity_label": current_entity["label"],
                "start_char": current_entity["start"],
                "end_char": current_entity["end"]
            })
    
    return pd.DataFrame(rows)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label → count for spaCy
          'hf_counts': dict of entity_label → count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
    """
    # TODO: Count entities per label for each system, compute totals
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()
    
    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": len(spacy_df),
        "total_hf": len(hf_df)
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0–1).
    """
    # TODO: Match predicted entities to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    # Create sets of (text_id, entity_text, entity_label) tuples
    predicted_set = set(
        zip(predicted_df["text_id"], predicted_df["entity_text"], predicted_df["entity_label"])
    )
    gold_set = set(
        zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"])
    )
    
    # Compute true positives, false positives, false negatives
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    
    # Compute precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    # Load data
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} articles, {df['language'].value_counts().to_dict()}")

        # Filter English texts
        en_df = df[df["language"] == "en"]
        texts = list(zip(en_df["id"], en_df["text"]))
        print(f"Processing {len(texts)} English articles")

        # Preprocess sample
        sample = preprocess_text(en_df["text"].iloc[0])
        if sample is not None:
            print(f"Sample preprocessed tokens: {sample[:10]}")

        # spaCy NER
        spacy_entities = extract_spacy_entities(texts)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER
        hf_entities = extract_hf_entities(texts)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nComparison: {comparison}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
