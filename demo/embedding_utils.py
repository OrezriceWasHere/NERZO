"""Utility functions for embedding text with the demo LLM+MLP stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from huggingface_hub import login
from tqdm import tqdm
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
        # gate = Gate(args.input_layer) if args.enable_gate else torch.nn.Identity()
        activation = self._build_activation(args.activation)
        noise = self._build_noise(args)
        middle = self._build_middle_layer(args.input_layer, args)
        output = self._build_output_layer(args.input_layer, args)

        self.net = torch.nn.Sequential(middle, activation, noise, output)

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


def load_llm_and_mlp(device: torch.device):
    """Initialise tokenizer, Llama model and MLP head with hook registered."""
    mlp_args = MLPArgs()
    mlp = MLP(mlp_args).to(device)
    mlp.load_state_dict(torch.load("contrastive_projection_head.pth", map_location=device))
    mlp = mlp.half()
    mlp.eval()

    # Login for HF gated models
    login()

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        output_hidden_states=False,
        torch_dtype=torch.float16,
    ).to(device).eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Register hook on layer-17 v_proj
    v_proj = model.model.layers[17].self_attn.v_proj
    v_proj.register_forward_hook(hooked_layer_function)

    return tokenizer, model, mlp


def embed_texts(texts: Dict[str, str], batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Embed id->text by treating each full text as a single span and using the end token.

    This is implemented via embed_entities_dataset to keep a single forward path.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, mlp = load_llm_and_mlp(device)

    # Wrap as FewNERD-style records with one span covering the whole text
    sentences = [
        {"id": k, "sentence": v, "predicted": [{"start": 0, "end": len(v)}]}
        for k, v in texts.items()
    ]

    records = embed_entities_dataset(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        mlp=mlp,
        device=device,
        batch_size=batch_size,
    )

    # Convert list-of-one to single tensor for backward compatibility
    out: Dict[str, torch.Tensor] = {}
    for k, v in records.items():
        if not v:
            continue
        out[k] = torch.tensor(v[0])
    return out

def embed_entities_batch(
    batch,
    tokenizer,
    model,
    mlp: torch.nn.Module,
    device: torch.device,
):
    """Embed entities for a batch of sentence records.

    Each record is expected to be a dict with at least:
      - id: unique identifier
      - sentence: raw text
      - predicted: list of spans, each having keys "start" and "end" (char indices)

    The representation is taken from the Llama 3.1 layer-17 v_proj at the token
    whose character span contains the entity's end offset, then projected by the MLP.

    Returns a mapping: id -> list of embedding vectors (as Python lists of floats).
    """

    model.eval()

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

    result = {}
    if "output" not in cache:
        return result

    vproj_batch = cache["output"]  # (B, L, H)

    def _token_indices_from_offsets(offsets_list, start_idx: int, end_idx: int):
        first_token_idx, last_token_idx = None, None
        for i, (token_start, token_end) in enumerate(offsets_list):
            if token_start <= start_idx < token_end and first_token_idx is None:
                first_token_idx = i
            if token_start < end_idx <= token_end:
                last_token_idx = i
                break
        if first_token_idx is None or last_token_idx is None:
            raise ValueError(
                f"Could not map text span ({start_idx}, {end_idx}) to token indices"
            )
        return first_token_idx, last_token_idx

    for b_idx, sent in enumerate(batch):
        vproj = vproj_batch[b_idx]
        offsets_b = offsets[b_idx].tolist()

        embs = []
        for ent in sent.get("predicted", []):
            if ent.get("end") == ent.get("start"):
                continue
            try:
                _, last_tok_index = _token_indices_from_offsets(offsets_b, ent["start"], ent["end"])
            except Exception:
                continue
            # Use the end token representation only (choose_llm_representation = 'end')
            end_token = vproj[last_tok_index]
            embs.append(mlp(end_token).detach().cpu().tolist())

        if embs:
            result[str(sent["id"])] = embs

    return result


def embed_entities_dataset(
    sentences,
    tokenizer,
    model,
    mlp: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
):
    """Embed entities for a full dataset by batching and calling embed_entities_batch."""

    all_results: Dict[str, list] = {}
    for start in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[start : start + batch_size]
        batch_result = embed_entities_batch(batch, tokenizer, model, mlp, device)
        for k, v in batch_result.items():
            all_results.setdefault(k, []).extend(v)
    return all_results
