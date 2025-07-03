from typing import List, Dict, Set, Tuple
import re

ENTITY_REGEX = re.compile(r"##(.*?)##")

########################################################################
#  ⇢  GOLD-SPAN RECONSTRUCTION  (keep fine-type)                       #
########################################################################

def gold_spans(
    tokens:     List[str],
    ner_tags:   List[int],
    fine_tags:  List[int],
    fine_id2lab: List[str] | None = None
) -> Tuple[List[Dict[str, str]], Set[str]]:
    """
    Any contiguous run of *identical* non-zero ner_tag IDs is considered one gold mention.
    We keep:
        1. a list of dicts     [{'text': ..., 'fine_type': ...}, ...]
        2. a set of strings    (mention texts) for span-level precision/recall.
    """
    spans: List[Dict[str, str]] = []
    current: List[str] = []
    cur_fine = 0
    prev_tag = 0

    def flush():
        if current:
            spans.append({
                "text": " ".join(current),
                "fine_type": fine_id2lab[cur_fine] if fine_id2lab else cur_fine
            })

    for word, tag, ftag in zip(tokens, ner_tags, fine_tags):
        if tag:
            if tag == prev_tag:
                current.append(word)
            else:
                flush()
                current  = [word]
                cur_fine = ftag
            prev_tag = tag
        else:
            flush()
            current = []
            prev_tag = 0
    flush()

    return spans, {d["text"] for d in spans}

########################################################################
#  ⇢  ENTITY EXTRACTION WITH POSITIONS (author’s logic)                #
########################################################################

def extract_entities_with_positions(sentence: str, response: str) -> List[Dict]:
    """Return list of extracted entities with their character offsets."""
    entities = []
    for match in ENTITY_REGEX.finditer(response):
        entity_text = match.group(1)
        start_idx = sentence.find(entity_text)
        if start_idx != -1:
            end_idx = start_idx + len(entity_text)
            entities.append({"text": entity_text, "start": start_idx, "end": end_idx})
        else:
            for m in re.finditer(re.escape(entity_text), sentence):
                entities.append({"text": entity_text,
                                 "start": m.start(), "end": m.end()})
                break
    return entities

########################################################################
#  ⇢  PREDICT + POST-PROCESS                                           #
########################################################################

def extract_entities(
    sentence: str,
    model,
    tokenizer,
    max_new_tokens: int = 256
) -> Tuple[Set[str], List[Dict]]:
    """
    Run CascadeNER extractor; return
        1) set of mention strings (for span-score),
        2) list of dicts  {'text', 'start', 'end'} (for offsets).
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    raw  = re.findall(r"##(.*?)##", text)
    span_set = {s.strip() for s in raw if s.strip()}
    span_pos = extract_entities_with_positions(sentence, text)
    return span_set, span_pos
