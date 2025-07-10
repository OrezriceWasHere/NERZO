import torch


def concat_llm_and_eos(llm_reprs: list[torch.Tensor], eos_vecs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Concatenate LLM final layer representations and EOS vectors on the same device."""
    return torch.stack([
        torch.cat((r.to(device), e.to(device)), dim=-1)
        for r, e in zip(llm_reprs, eos_vecs)
    ])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_r = [torch.randn(1, 10) for _ in range(2)]
    dummy_e = [torch.randn(1, 5) for _ in range(2)]
    result = concat_llm_and_eos(dummy_r, dummy_e, device)
    print(result.shape)
