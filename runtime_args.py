from dataclasses import dataclass

from typing import Optional
from os import environ as env
from functools import cache

from pydantic import BaseModel, Field


@cache
def is_key_enabled_in_env(key):
    return True if env.get(key) == "yes" else False

@cache
def get_env(key, default_value=None):
    return env.get(key, default_value)

@dataclass
class RuntimeArgs:
    compute_queue: str = "a100_gpu"
    debug_llm: bool = False
    max_llm_layer: Optional[int] = None
    upload_all_predictions: bool = False
    upload_model: bool = True
    allow_clearml: bool = is_key_enabled_in_env("ALLOW_CLEARML")
    running_remote: bool = is_key_enabled_in_env("RUNNING_REMOTE")

class ElasticsearchConnection(BaseModel):
    hosts: list[str] = Field(default_factory=lambda:get_env("ELASTICSEARCH_HOSTS", "http://dsigpu06:9200,http://dsigpu08:9200,http://dsigpu07:9200").split(","))
    verify_certs: bool = False
    request_timeout: int = 270
    ssl_show_warn: bool = False
    max_retries: int = 10
    retry_on_timeout: bool = True
    basic_auth: tuple[str, str] = Field(default_factory=lambda:
    (get_env("ELASTICSEARCH_USER", "elastic"), get_env("ELASTICSEARCH_PASSWORD", "XXX")))