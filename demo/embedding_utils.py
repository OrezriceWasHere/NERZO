"""Utility functions for embedding text with the demo LLM+MLP stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from huggingface_hub import login
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


def load_llm_and_mlp(device: torch.device):
    """Initialise tokenizer, Llama model and MLP head with hook registered."""
    mlp_args = MLPArgs()
    mlp = MLP(mlp_args).to(device)
    mlp.load_state_dict(torch.load("contrastive_projection_head.pth", map_location=device))
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
    handle = v_proj.register_forward_hook(hooked_layer_function)

    return tokenizer, model, mlp, handle


def embed_text_mapping(
    texts: Dict[str, str],
    tokenizer,
    model,
    mlp: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """Embed a mapping of id->text using the last token's representation."""

    model.eval()
    keys = list(texts.keys())
    values = [texts[k] for k in keys]
    embeddings: Dict[str, torch.Tensor] = {}

    for start in range(0, len(values), batch_size):
        batch_texts = values[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
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
        for b_idx, key in enumerate(keys[start : start + batch_size]):
            token_vec = vproj_batch[b_idx, -1]
            embeddings[key] = mlp(token_vec).detach().cpu()

    return embeddings


def embed_texts(texts: Dict[str, str], batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Embed a mapping of identifier -> text using Llama 3.1 and the MLP head."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, mlp, handle = load_llm_and_mlp(device)
    try:
        return embed_text_mapping(texts, tokenizer, model, mlp, device, batch_size=batch_size)
    finally:
        handle.remove()

