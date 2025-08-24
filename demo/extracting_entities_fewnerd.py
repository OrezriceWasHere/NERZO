"""Extract entities from the full FewNERD dataset and evaluate fuzzy span recall.

This script downloads the complete FewNERD dataset, predicts entity spans for
each sentence with the ``CascadeNER/models_for_CascadeNER`` language model and
stores the results in a JSON file.  The helper functions that interface with
this model live in ``cascade_llm_entity_extractor.py`` so that the extraction
logic is isolated from the dataset handling shown here.

The output JSON is a list of records with the following structure::

    {
        "id": "<uuid>",
        "sentence": "Barack Obama visited Paris in 2015.",
        "gold": [
            {"text": "Barack Obama", "start": 0, "end": 12, "label": "person/actor"},
            {"text": "Paris", "start": 20, "end": 25, "label": "location/city"}
        ],
        "predicted": [
            {"text": "Barack Obama", "start": 0, "end": 12},
            {"text": "Paris", "start": 20, "end": 25}
        ]
    }

Run the script with::

    python extracting_entities_fewnerd.py --output fewnerd_entities.json

Use ``--limit`` to process only the first ``N`` sentences for quicker demos.
"""

from __future__ import annotations

import argparse
import itertools
import json
import uuid
from typing import List, Sequence, Tuple

from datasets import load_dataset

from cascade_llm_entity_extractor import load_cascadener, predict_spans
from fuzzy_span_recall import count_fuzzy_matches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tags_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO tags to character spans."""
    spans: List[Tuple[int, int, str]] = []
    current_label: str | None = None
    start_char: int | None = None

    offset = 0
    for token, tag in zip(tokens, tags):
        token_start = offset
        token_end = token_start + len(token)
        offset = token_end + 1  # account for joining spaces

        if tag.startswith("B-"):
            if current_label is not None:
                spans.append((start_char, prev_end, current_label))
            current_label = tag[2:]
            start_char = token_start
        elif tag.startswith("I-") and current_label == tag[2:]:
            pass
        else:
            if current_label is not None:
                spans.append((start_char, prev_end, current_label))
                current_label = None
        prev_end = token_end

    if current_label is not None:
        spans.append((start_char, prev_end, current_label))
    return spans


# Example FewNERD slice kept only for reference; it is not used by the script.
# SAMPLE_DATASET = [
#     {
#         "tokens": ["Barack", "Obama", "visited", "Paris", "in", "2015", "."],
#         "ner_tags": [
#             "B-person/actor",
#             "I-person/actor",
#             "O",
#             "B-location/city",
#             "O",
#             "O",
#             "O",
#         ],
#     }
# ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="fewnerd_entities.json",
        help="Path to write extracted entity dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sentences (for quick demos)",
    )
    args = parser.parse_args()

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    label_names = dataset["train"].features["ner_tags"].feature.names

    tokenizer, model = load_cascadener()

    records: List[dict] = []
    total_gold = 0
    total_matched = 0

    all_examples = itertools.chain(
        dataset["train"], dataset["validation"], dataset["test"]
    )
    for example in itertools.islice(all_examples, args.limit):
        tokens = example["tokens"]
        text = " ".join(tokens)
        tags = [label_names[t] for t in example["ner_tags"]]
        gold_spans = tags_to_spans(tokens, tags)

        pred_spans = predict_spans(text, tokenizer, model)

        gold_texts = [text[start:end] for start, end, _ in gold_spans]
        pred_texts = [p["text"] for p in pred_spans]
        matched, gold_count = count_fuzzy_matches(gold_texts, pred_texts)
        total_matched += matched
        total_gold += gold_count

        record = {
            "id": str(uuid.uuid4()),
            "sentence": text,
            "gold": [
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "label": label,
                }
                for start, end, label in gold_spans
            ],
            "predicted": pred_spans,
        }
        records.append(record)

    recall = total_matched / total_gold if total_gold else 0.0

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} sentences to {args.output}")
    print(f"Span recall: {recall:.3f}")


if __name__ == "__main__":
    main()

