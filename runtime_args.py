from dataclasses import dataclass
from typing import Optional
from os import environ as env
from functools import cache

@cache
def is_key_enabled_in_env(key):
    return True if env.get(key) == "yes" else False


@dataclass
class RuntimeArgs:
    compute_queue: str = "a100_gpu"
    debug_llm: bool = False
    max_llm_layer: Optional[int] = None
    upload_all_predictions: bool = False
    upload_model: bool = True
    allow_clearml: bool = is_key_enabled_in_env("ALLOW_CLEARML")
    running_remote: bool = is_key_enabled_in_env("RUNNING_REMOTE")
