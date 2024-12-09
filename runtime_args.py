from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeArgs:
    compute_queue: str = "gpu"
    debug_llm: bool = False
    max_llm_layer: Optional[int] = 18