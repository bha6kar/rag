import os
from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEV_LLM_CONFIG_PATH = BASE_DIR / "config" / "llm-config.yml"

MODEL_CONFIG_KEYS = [
    "project_id",
    "location",
    "model_name",
    "endpoint_id",
    "embedding_model_name",
]
GENERATION_CONFIG_KEYS = ["temperature", "max_output_tokens", "top_p", "top_k"]
RAG_KEYS = ["vectordb_path"]

DEFAULT_CONFIG = {
    "model": {},
    "generation": {},
    "rag": {},
}


def get_llm_config() -> dict:
    """Fetches the LLM configuration from a local YAML file."""
    config = _read_local_config()

    env_project_id = os.getenv("GCP_PROJECT")
    if env_project_id:
        config["project_id"] = env_project_id

    return {
        "model": {key: config.get(key, None) for key in MODEL_CONFIG_KEYS},
        "generation": {key: config.get(key, None) for key in GENERATION_CONFIG_KEYS},
        "rag": {key: config.get(key, None) for key in RAG_KEYS},
    }


def _read_local_config() -> dict:
    try:
        with open(DEV_LLM_CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        logger.error(f"LLM config file not found: {DEV_LLM_CONFIG_PATH}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    return DEFAULT_CONFIG
