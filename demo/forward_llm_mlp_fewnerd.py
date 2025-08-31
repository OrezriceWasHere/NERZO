"""Forward FewNERD entities through Llama 3.1 and an MLP head.

This script reads the sentence-level JSON produced by
``extracting_entities_fewnerd.py`` and generates an embedding for every gold
entity span.  The embedding is taken from block 17 of ``Meta‑Llama‑3.1`` and
projected through an MLP whose architecture mirrors the training-time model in
``mlp.py``.  The resulting vectors are written as a binary ``.pth`` file mapping
each sentence identifier to a list of its entity embeddings.

Run the script with::

    python forward_llm_mlp_fewnerd.py \
        --input fewnerd_entities.json --output fewnerd_embeddings.pth
"""
from __future__ import annotations
from tqdm.auto import trange  # progress bar
import argparse
import json
from typing import Dict, List
import torch

from embedding_utils import load_llm_and_mlp, cache


def embed_with_llama(
		sentences: List[Dict],
		tokenizer,
		model,
		mlp: torch.nn.Module,
		device: torch.device,
		batch_size: int = 8,
) -> Dict[str, List[List[float]]]:
	"""Embed entities using Llama 3.1 layer-17 V-projection, batched."""

	model.eval()
	records: Dict[str, List[List[float]]] = {}

	for start in trange(0, len(sentences), batch_size, desc="Batches"):
		batch = sentences[start: start + batch_size]
		texts = [s["sentence"] for s in batch]

		tokens = tokenizer(
			texts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			return_offsets_mapping=True,
			add_special_tokens=True,
		)
		offsets = tokens.pop("offset_mapping")
		tokens = {k: v.to(device) for k, v in tokens.items()}

		with torch.no_grad():
			cache.clear()
			_ = model(
				input_ids=tokens["input_ids"],
				attention_mask=tokens["attention_mask"],
				output_hidden_states=False,
				use_cache=False,
			)

		if "output" not in cache:
			continue

		vproj_batch = cache["output"]
		for b_idx, sent in enumerate(batch):
			vproj = vproj_batch[b_idx]
			offsets_b = offsets[b_idx].tolist()

			embs: List[List[float]] = []
			for ent in sent.get("predicted", []):
				end_tok_index = next((i for i, (s, e) in enumerate(offsets_b) if s < ent["end"] <= e), None)
				if end_tok_index is None:
					continue
				end_token = vproj[end_tok_index]
				embs.append(mlp(end_token).detach().cpu().tolist())

			if embs:
				records[str(sent["id"])] = embs

	return records


def main() -> None:
	parser = argparse.ArgumentParser(description="Embed entities from Llama 3.1 layer-17 v_proj")
	parser.add_argument("--input", default="fewnerd_entities.json", help="Input JSON file")
	parser.add_argument("--output", default="fewnerd_embeddings.pth", help="Output .pth file")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size for model forward")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer, model, mlp = load_llm_and_mlp(device)

	with open(args.input, "r", encoding="utf8") as f:
		sentences = json.load(f)

	records = embed_with_llama(
		sentences=sentences,
		tokenizer=tokenizer,
		model=model,
		mlp=mlp,
		device=device,
		batch_size=args.batch_size,
	)

	torch.save(records, args.output)
	print(
		f"Wrote embeddings for {sum(len(v) for v in records.values())} entities to {args.output}"
	)


if __name__ == "__main__":
	main()
