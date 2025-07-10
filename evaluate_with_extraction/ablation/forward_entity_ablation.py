import torch


def concat_llm_and_eos(llm_reprs: list[torch.Tensor], eos_vecs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Concatenate LLM representations and EOS vectors along the last dimension
    while ensuring everything is moved to the same device before stacking.
    """
    concat_pairs = []
    for r, e in zip(llm_reprs, eos_vecs):
        r = r.to(device)
        e = e.to(device)
        concat_pairs.append(torch.cat((r, e), dim=-1))
    return torch.stack(concat_pairs)


# Example usage within process_batch (not executed on import)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_r = [torch.randn(1, 10) for _ in range(2)]
    dummy_e = [torch.randn(1, 5) for _ in range(2)]
    result = concat_llm_and_eos(dummy_r, dummy_e, device)
    print(result.shape)
