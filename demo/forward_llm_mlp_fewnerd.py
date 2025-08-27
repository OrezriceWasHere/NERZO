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
from tqdm.auto import trange  # add this import
from huggingface_hub import login
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



class Gate(torch.nn.Module):
    """Per-dimension gating module copied from ``mlp.py``."""

    def __init__(self, size: int):
        super().__init__()
        self.dimension = size
        self.gate = torch.nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return x * torch.sigmoid(self.gate)

    def extra_repr(self) -> str:  # pragma: no cover - diagnostic
        return f"dimension={self.dimension}"


@dataclass
class MLPArgs:
    input_layer: int = 1024
    hidden_layer: int = 500
    output_layer: int = 500
    enable_gate: bool = True
    activation: str = "silu"
    noise: str = "dropout"
    is_hidden_layer: bool = True
    dropout: float = 0.1


class MLP(torch.nn.Module):
    """MLP head matching the training configuration in ``mlp.py``."""

    def __init__(self, args: MLPArgs):
        super().__init__()
        gate = Gate(args.input_layer) if args.enable_gate else torch.nn.Identity()
        activation = self._build_activation(args.activation)
        noise = self._build_noise(args)
        middle = self._build_middle_layer(args.input_layer, args)
        output = self._build_output_layer(args.input_layer, args)

        self.net = torch.nn.Sequential(gate, middle, activation, noise, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)

    @staticmethod
    def _build_activation(name: str) -> torch.nn.Module:
        if name == "silu":
            return torch.nn.SiLU()
        if name == "leaky_relu":
            return torch.nn.LeakyReLU()
        return torch.nn.ReLU()

    @staticmethod
    def _build_noise(args: MLPArgs) -> torch.nn.Module:
        if args.noise == "dropout":
            return torch.nn.Dropout(args.dropout)
        return torch.nn.Identity()

    @staticmethod
    def _build_middle_layer(input_layer: int, args: MLPArgs) -> torch.nn.Module:
        if args.is_hidden_layer:
            return torch.nn.Linear(input_layer, args.hidden_layer)
        return torch.nn.Identity()

    @staticmethod
    def _build_output_layer(input_layer: int, args: MLPArgs) -> torch.nn.Module:
        if args.is_hidden_layer:
            return torch.nn.Linear(args.hidden_layer, args.output_layer)
        return torch.nn.Linear(input_layer, args.output_layer)

# ---- hook cache ----
cache: Dict[str, torch.Tensor] = {}

def hooked_layer_function(_mod, _inp, out):
    # out: (B, L, hidden_size)
    cache.clear()
    cache["output"] = out.detach()

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

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for start in trange(0, len(sentences), batch_size, desc="Batches"):
        batch = sentences[start : start + batch_size]
        texts = [s["sentence"] for s in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            cache.clear()
            _ = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
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
                end_tok_index   = next((i for i, (s,e) in enumerate(offsets_b) if s <  ent["end"]   <= e), None)
                if  end_tok_index is None:
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

    # Your MLP and weights (unchanged)
    mlp_args = MLPArgs()
    mlp = MLP(mlp_args).to(device)
    mlp.load_state_dict(torch.load("contrastive_projection_head.pth", map_location=device))
    mlp.eval()

    with open(args.input, "r", encoding="utf8") as f:
        sentences = json.load(f)

    login()  # unchanged

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        use_fast=True,
    )
    # Minimal fix: EOS as PAD so batching works
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        output_hidden_states=False,   # not needed when using hook
        torch_dtype=torch.float16,
    ).to(device).eval()
    # Make model aware of PAD id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Register hook once on layer-17 v_proj (0-based index)
    v_proj = model.model.layers[17].self_attn.v_proj
    handle = v_proj.register_forward_hook(hooked_layer_function)

    try:
        records = embed_with_llama(
            sentences=sentences,
            tokenizer=tokenizer,
            model=model,
            mlp=mlp,
            device=device,
            batch_size=args.batch_size,
        )
    finally:
        handle.remove()  # always clean up

    torch.save(records, args.output)
    print(f"Wrote embeddings for {sum(len(v) for v in records.values())} entities to {args.output}")

if __name__ == "__main__":
    main()
