#!/usr/bin/env python3
"""
Evaluate entity extraction with exact-match precision, recall, and F1.

Usage:
    python eval_entities.py results.json
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from extract_entities_regex_v2 import align_to_original

import re
import unicodedata
from typing import Set

STOP_DETS = {"the", "a", "an"}

def normalise(text: str) -> str:
    """Lower-case, strip accents & leading determiners."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    tokens = re.split(r"\W+", text.lower().strip())
    if tokens and tokens[0] in STOP_DETS:
        tokens = tokens[1:]
    return " ".join(tokens)

def jaccard(a, b):            # token sets
    return len(a & b) / len(a | b) if (a | b) else 0.0

def is_match(gold: str, pred: str, min_jacc=0.8) -> bool:
    gold_t, pred_t = set(gold.split()), set(pred.split())
    return (
        gold_t <= pred_t                                     # containment
        or pred_t <= gold_t
        or jaccard(gold_t, pred_t) >= min_jacc               # high overlap
    )

def get_entity_set(marked: str, original: str) -> Set[str]:
    """
    Extract entities from a text and return them as a set of lowercased strings.
    """
    entities = align_to_original(marked,original, longest_only=True,all_occurrences=True)
    entities_text = [e['text'] for e in entities if e['text'] is not None]
    return entities_text, entities



def evaluate(json_path: Path, min_jacc: float = 0.8) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # exact counters
    tp_e = fp_e = fn_e = 0
    # fuzzy counters
    tp_f = fp_f = fn_f = 0

    for record in tqdm(data.values()):
        gold_raw = [g["text"] for g in record["gold"] if g["text"]]
        pred_raw, pred_with_positions = get_entity_set(marked=record["generated"],original=record["sentence"])      # → list / iterable

        gold_norm = list({normalise(t) for t in gold_raw})
        pred_norm = list({normalise(t) for t in pred_raw})

        # ----- Exact --------------------------------------------------------
        gold_set = set(gold_norm)
        pred_set = set(pred_norm)

        tp_e += len(gold_set & pred_set)
        fp_e += len(pred_set - gold_set)
        fn_e += len(gold_set - pred_set)


        # if  len(gold_set - pred_set) > 0:
        #     print(f"Record {record['id']} has missing entities: {gold_set - pred_set}")
        #     #breakpoint()
        # ----- Fuzzy (greedy pairing) ---------------------------------------
        matched_gold = set()
        matched_pred = set()

        for gi, g in enumerate(gold_norm):
            for pj, p in enumerate(pred_norm):
                if pj in matched_pred:
                    continue
                if is_match(g, p, min_jacc):
                    matched_gold.add(gi)
                    matched_pred.add(pj)
                    break

        tp_f += len(matched_gold)
        fp_f += len(pred_norm) - len(matched_pred)
        fn_f += len(gold_norm) - len(matched_gold)

        # if len(gold_norm) - len(matched_gold) > 0:
        #     print(f"Record {record['id']} has missing entities: {set(gold_norm) - pred_set}")


    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    p_e, r_e, f1_e = prf(tp_e, fp_e, fn_e)
    p_f, r_f, f1_f = prf(tp_f, fp_f, fn_f)

    print("=== Exact match ===")
    print(f"TP {tp_e}  FP {fp_e}  FN {fn_e}")
    print(f"Precision {p_e:.4f}  Recall {r_e:.4f}  F1 {f1_e:.4f}\n")

    print("=== Fuzzy match (min Jaccard ≥ {min_jacc}) ===")
    print(f"TP {tp_f}  FP {fp_f}  FN {fn_f}")
    print(f"Precision {p_f:.4f}  Recall {r_f:.4f}  F1 {f1_f:.4f}")

if __name__ == "__main__":
    file_path = '/home/orsh/.clearml/cache/storage_manager/datasets/ds_ce91b3feb74348cc9261959dac2d5db8/nertrieve_test_ir_base_combined.json'
    evaluate(file_path)