# Base extraction logic for LLM span extraction
import gc
import hashlib
import json
import re
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from clearml import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ENTITY_REGEX = re.compile(r"##(.*?)##")


def build_prompt(sentence: str, tokenizer: AutoTokenizer) -> str:
    """Create a chat prompt from a sentence."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sentence},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Helper functions used by the datasets
# ---------------------------------------------------------------------------

def extract_entities_with_positions(sentence: str, response: str) -> List[Dict]:
    """Return list of {'text','start','end'} for every ##…## chunk."""
    entities: List[Dict[str, int | str]] = []
    for m in ENTITY_REGEX.finditer(response):
        ent_text = m.group(1)
        start = sentence.find(ent_text)
        if start != -1:
            entities.append({"text": ent_text, "start": start, "end": start + len(ent_text)})
        else:
            for m2 in re.finditer(re.escape(ent_text), sentence):
                entities.append({"text": ent_text, "start": m2.start(), "end": m2.end()})
                break
    return entities


# Functions below are copied from the FewNERD extraction script
# to align generated text with the original sentence.

def _is_open(text: str, i: int, inside: bool) -> bool:
    if not inside:
        return True
    nxt = i + 2
    return nxt < len(text) and text[nxt].isalnum()


def extract_entities(text: str, longest_only: bool = False) -> List[Tuple[str, int, int]]:
    clean: List[str] = []
    clean_pos = 0
    i = 0
    stack: List[int] = []
    entities: List[Tuple[str, int, int]] = []

    while i < len(text):
        if text.startswith("##", i):
            if _is_open(text, i, inside=bool(stack)):
                stack.append(clean_pos)
            elif stack:
                start = stack.pop()
                entities.append(("".join(clean[start:clean_pos]), start, clean_pos))
            i += 2
            continue

        clean.append(text[i])
        clean_pos += 1
        i += 1

    if longest_only:
        outer: List[Tuple[str, int, int]] = []
        for t, s, e in sorted(entities, key=lambda x: x[2] - x[1], reverse=True):
            if not any(s >= os and e <= oe for _, os, oe in outer):
                outer.append((t, s, e))
        return outer

    return entities


def _find_next(hay: str, needle: str, start: int) -> Tuple[int, int]:
    k = hay.find(needle, start)
    return (k, k + len(needle)) if k != -1 else (-1, -1)


def _remove_nested(spans: List[Dict]) -> List[Dict]:
    keep: List[Dict] = []
    for sp in sorted(spans, key=lambda d: d['end'] - d['start'], reverse=True):
        if not any(sp['start'] >= k['start'] and sp['end'] <= k['end'] for k in keep):
            keep.append(sp)
    return sorted(keep, key=lambda d: d['start'])


def align_to_original(marked: str, original: str, longest_only: bool = True, all_occurrences: bool = True) -> List[Dict]:
    ents = extract_entities(marked, longest_only)
    out: List[Dict] = []
    cur = 0

    for text, _, _ in ents:
        if all_occurrences:
            for m in re.finditer(re.escape(text), original):
                out.append({"text": text, "start": m.start(), "end": m.end()})
        else:
            b, e = _find_next(original, text, cur)
            if b != -1:
                out.append({"text": text, "start": b, "end": e})
                cur = e

    return _remove_nested(out)


class LLMExtractionRunner:
    """Generic span extraction runner using a causal LLM."""

    def __init__(
        self,
        sentences: Sequence[str],
        gold_lists: Sequence[Sequence[Dict]],
        gold_sets: Sequence[set],
        ids: Sequence[str] | None,
        extract_fn: Callable[[str, str], List[Dict]],
        batch_size: int = 1,
        max_new_tokens: int = 256,
        repo: str = "CascadeNER/models_for_CascadeNER",
        subfolder: str = "extractor",
        dataset_project: str = "pipeline",
        results_path: str = "span_extraction_results.json",
    ) -> None:
        self.sentences = list(sentences)
        self.gold_lists = list(gold_lists)
        self.gold_sets = list(gold_sets)
        self.ids = list(ids) if ids else [None] * len(self.sentences)
        self.extract_fn = extract_fn
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.results_path = results_path
        self.dataset_project = dataset_project

        self.tokenizer = AutoTokenizer.from_pretrained(
            repo, subfolder=subfolder, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def evaluate(self) -> None:
        prompts = [build_prompt(s, self.tokenizer) for s in self.sentences]
        gold_total = pred_total = correct = 0
        results: Dict[str, Dict] = {}
        pbar = tqdm(range(0, len(prompts), self.batch_size))

        with torch.no_grad():
            for start in pbar:
                b_prompts = prompts[start : start + self.batch_size]
                b_sents = self.sentences[start : start + self.batch_size]
                b_gold_l = self.gold_lists[start : start + self.batch_size]
                b_gold_set = self.gold_sets[start : start + self.batch_size]

                enc = self.tokenizer(b_prompts, return_tensors="pt", padding=True).to(self.model.device)
                prompt_len = enc["attention_mask"].sum(1)

                outs = self.model.generate(**enc, max_new_tokens=self.max_new_tokens)
                decoded = self.tokenizer.batch_decode(
                    [seq[prompt_len[i] :] for i, seq in enumerate(outs)], skip_special_tokens=True
                )

                for i, generated in enumerate(decoded):
                    preds_pos = self.extract_fn(b_sents[i], generated)
                    for p in preds_pos:
                        assert b_sents[i][p["start"] : p["end"]] == p["text"]
                    preds_txt = {p["text"].strip() for p in preds_pos if p["text"].strip()}

                    gold_total += len(b_gold_set[i])
                    pred_total += len(preds_txt)
                    correct += len(b_gold_set[i] & preds_txt)

                    sent_hash = hashlib.sha1(b_sents[i].encode()).hexdigest()
                    results[sent_hash] = {
                        "id": self.ids[start + i],
                        "sentence": b_sents[i],
                        "gold": b_gold_l[i],
                        "predicted": preds_pos,
                        "generated": generated,
                    }
                del enc
                del outs
                del decoded
                torch.cuda.empty_cache()
                prec = correct / pred_total if pred_total else 0
                rec = correct / gold_total if gold_total else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
                pbar.set_description(f"Prec {prec:.2f}  Rec {rec:.2f}  F1 {f1:.2f}")

                torch.cuda.empty_cache()
                gc.collect()

        print("\n— Final results —")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-score:  {f1:.3f}")

        with open(self.results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

        cl_ds = Dataset.create(dataset_name=self.results_path, dataset_project=self.dataset_project)
        cl_ds.add_files(path=self.results_path)
        cl_ds.add_tags([self.results_path])
        cl_ds.upload()
        cl_ds.finalize()
