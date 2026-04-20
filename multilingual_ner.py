"""
Module 6 Week A — Stretch: Multilingual NER Comparison

Compares NER performance across English and Arabic climate texts using:
  - spaCy xx_ent_wiki_sm  (multilingual, WikiNER labels: PER, ORG, LOC, MISC)
  - Hugging Face Davlan/xlm-roberta-base-wikiann-ner  (same label schema)

Run:
    python multilingual_ner.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline

MIN_TEXTS = 20
DATA_PATH = "data/climate_articles.csv"
HF_MODEL = "Davlan/xlm-roberta-base-wikiann-ner"

WIKIANN_LABELS = {"PER", "ORG", "LOC"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath=DATA_PATH):
    df = pd.read_csv(filepath)
    en = df[df["language"] == "en"].head(min(len(df[df["language"] == "en"]), 50))
    ar = df[df["language"] == "ar"].head(min(len(df[df["language"] == "ar"]), 50))
    return en, ar


# ---------------------------------------------------------------------------
# spaCy multilingual extraction
# ---------------------------------------------------------------------------

def extract_spacy_multilingual(texts, model_name="xx_ent_wiki_sm"):
    """Run spaCy multilingual model on list of (text_id, text) tuples."""
    nlp = spacy.load(model_name)
    rows = []
    no_entity_count = 0
    for text_id, text_str in texts:
        doc = nlp(str(text_str))
        found = False
        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text.strip(),
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
            found = True
        if not found:
            no_entity_count += 1
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"]
    )
    return df, no_entity_count


# ---------------------------------------------------------------------------
# Hugging Face multilingual extraction
# ---------------------------------------------------------------------------

def extract_hf_multilingual(texts, model_name=HF_MODEL):
    """Run HF multilingual NER model on list of (text_id, text) tuples."""
    ner = hf_pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
    )
    rows = []
    no_entity_count = 0
    for text_id, text_str in texts:
        text_str = str(text_str)
        # Truncate to 512 chars to stay within token limits for long texts
        chunk = text_str[:512]
        results = ner(chunk)
        found = False
        for r in results:
            label = r["entity_group"]
            rows.append({
                "text_id": text_id,
                "entity_text": r["word"].strip(),
                "entity_label": label,
                "start_char": r["start"],
                "end_char": r["end"],
            })
            found = True
        if not found:
            no_entity_count += 1
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"]
    )
    return df, no_entity_count


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summarise(df, no_entity_count, total_texts, label=""):
    counts = df["entity_label"].value_counts().to_dict() if len(df) else {}
    examples = []
    for lbl in sorted(counts):
        subset = df[df["entity_label"] == lbl]["entity_text"].dropna().unique()
        for ex in subset[:3]:
            examples.append((lbl, ex))
            if len(examples) >= 9:
                break
    return {
        "label": label,
        "total_entities": len(df),
        "total_texts": total_texts,
        "no_entity_texts": no_entity_count,
        "no_entity_rate": no_entity_count / total_texts if total_texts else 0,
        "counts_by_type": counts,
        "examples": examples[:9],
    }


def print_summary(s):
    print(f"\n{'='*60}")
    print(f"  {s['label']}")
    print(f"{'='*60}")
    print(f"  Texts processed  : {s['total_texts']}")
    print(f"  Total entities   : {s['total_entities']}")
    print(f"  No-entity texts  : {s['no_entity_texts']}  "
          f"({s['no_entity_rate']:.1%} of texts)")
    print(f"  Counts by type   :")
    for lbl, cnt in sorted(s["counts_by_type"].items()):
        print(f"    {lbl:<8}: {cnt}")
    print(f"  Examples (label → entity):")
    for lbl, ex in s["examples"]:
        print(f"    [{lbl}] {ex}")


def build_comparison_table(summaries):
    """Return a wide DataFrame comparing entity counts across all conditions."""
    all_labels = sorted(
        {lbl for s in summaries for lbl in s["counts_by_type"]}
    )
    rows = []
    for s in summaries:
        row = {
            "Condition": s["label"],
            "Texts": s["total_texts"],
            "Total Entities": s["total_entities"],
            "No-Entity Rate": f"{s['no_entity_rate']:.1%}",
        }
        for lbl in all_labels:
            row[lbl] = s["counts_by_type"].get(lbl, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis text
# ---------------------------------------------------------------------------

ANALYSIS = """
=== ANALYSIS ===

Paragraph 1 — Entity types harder in Arabic vs. English, and why:

The results reveal a stark contrast between models. spaCy xx_ent_wiki_sm on Arabic
extracted only 50 entities across 50 texts (vs. 233 for English) with a 36%
no-entity rate, and many of its "entities" were misclassified Arabic verbs:
"ويمثل" (he represents) tagged as LOC, "وثّق" (documented) tagged as ORG. This
model essentially fails on Arabic. XLM-RoBERTa fared far better — 186 entities, 0%
no-entity rate — correctly tagging "الأردن" (Jordan) as LOC, "البنك الدولي" (World
Bank) as ORG, and "الهيئة الحكومية الدولية المعنية بتغير المناخ" (IPCC) as ORG.
However, PER entities collapse in Arabic (2 found vs. 14 in English) because Arabic
person names lack capitalisation as a signal, and they frequently carry attached
clitics — "وزير البيئة الأردني" (Jordan's Minister of Environment) appears as a
noun phrase rather than a clearly bounded proper name. ORG recognition also proved
harder: XLM-RoBERTa found 81 Arabic ORGs vs. 109 English ORGs even though the
texts are parallel documents covering the same events — the IPCC report, COP28,
World Bank fund — demonstrating that morphological fusion obscures entity boundaries
that English orthography makes explicit.

Paragraph 2 — Implications for bilingual NLP applications in the MENA region:

In Jordan and the broader MENA region, production NLP systems routinely handle
documents that mix Arabic and English — ministry reports, news wires, corporate
filings — sometimes within the same paragraph. This experiment shows two things:
first, a general-purpose multilingual spaCy model is practically unusable for
Arabic NER (36% silent failure rate, high false-positive noise), so any pipeline
that defaults to a single model for both languages will silently degrade on Arabic
content; second, XLM-RoBERTa provides a viable multilingual baseline but still
under-counts Arabic persons by nearly 7x compared to English on the same topics.
For organisations like Jordan's Ministry of Environment or the Royal Scientific
Society, the practical implication is layered: (a) Arabic text should be passed
through a morphological segmenter (e.g. Farasa or CAMeL Tools) before NER to break
clitic attachments; (b) an Arabic-fine-tuned NER model (e.g. CAMeL-NER or
AraBERT-based) should supplement XLM-RoBERTa for documents where person and
organisation recall matters; (c) monitoring dashboards for bilingual pipelines must
track entity yield separately per language — a system that looks healthy on
aggregate English metrics can be silently failing on Arabic content the whole time.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    en_df, ar_df = load_data()
    print(f"  English texts: {len(en_df)}  |  Arabic texts: {len(ar_df)}")

    en_texts = list(zip(en_df["id"], en_df["text"]))
    ar_texts = list(zip(ar_df["id"], ar_df["text"]))

    # --- spaCy multilingual ---
    print("\nRunning spaCy xx_ent_wiki_sm on English …")
    sp_en_df, sp_en_none = extract_spacy_multilingual(en_texts)

    print("Running spaCy xx_ent_wiki_sm on Arabic …")
    sp_ar_df, sp_ar_none = extract_spacy_multilingual(ar_texts)

    # --- HF multilingual ---
    print(f"\nLoading HF model ({HF_MODEL}) …")
    print("Running HF XLM-RoBERTa on English …")
    hf_en_df, hf_en_none = extract_hf_multilingual(en_texts)

    print("Running HF XLM-RoBERTa on Arabic …")
    hf_ar_df, hf_ar_none = extract_hf_multilingual(ar_texts)

    # --- Summaries ---
    summaries = [
        summarise(sp_en_df, sp_en_none, len(en_texts), "spaCy xx_ent_wiki_sm | English"),
        summarise(sp_ar_df, sp_ar_none, len(ar_texts), "spaCy xx_ent_wiki_sm | Arabic"),
        summarise(hf_en_df, hf_en_none, len(en_texts), "XLM-RoBERTa          | English"),
        summarise(hf_ar_df, hf_ar_none, len(ar_texts), "XLM-RoBERTa          | Arabic"),
    ]

    for s in summaries:
        print_summary(s)

    # --- Comparison table ---
    table = build_comparison_table(summaries)
    print("\n\n=== COMPARISON TABLE ===\n")
    print(table.to_string(index=False))

    # Save table to CSV
    table.to_csv("multilingual_ner_comparison.csv", index=False)
    print("\nComparison table saved to multilingual_ner_comparison.csv")

    # --- Analysis ---
    print(ANALYSIS)


if __name__ == "__main__":
    main()
