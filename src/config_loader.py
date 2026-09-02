"""
config_loader.py

Loads config/config.yaml, resolves relative paths against the project
root, and pulls secrets (local LLM server key, SMTP password, admin auth)
from environment variables / a .env file. Nothing sensitive lives in the
YAML file itself.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """Load config.yaml, load .env, resolve paths, attach secrets."""
    load_dotenv(PROJECT_ROOT / ".env")  # no-op if the file doesn't exist

    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # resolve relative paths against the project root
    cfg["database"]["sqlite_path"] = str(PROJECT_ROOT / cfg["database"]["sqlite_path"])
    cfg["chroma"]["persist_directory"] = str(PROJECT_ROOT / cfg["chroma"]["persist_directory"])

    # attach secrets from environment
    llm_api_key_env = cfg["llm"]["api_key_env"]
    # most local OpenAI-compatible servers ignore the key value but still
    # require the Authorization header to be non-empty
    cfg["llm"]["api_key"] = os.environ.get(llm_api_key_env) or "not-needed"

    # HF Inference API token for the embedding backend (see
    # src/embeddings.py::HFInferenceEmbeddingFunction). Only required when
    # llm.embedding_backend is "hf_inference" (the default).
    hf_cfg = cfg["llm"].get("hf_inference") or {}
    hf_api_key_env = hf_cfg.get("api_key_env", "HF_API_TOKEN")
    cfg["llm"]["hf_api_key"] = os.environ.get(hf_api_key_env)

    smtp_pw_env = cfg["admin"]["email"]["smtp_password_env"]
    cfg["admin"]["email"]["smtp_password"] = os.environ.get(smtp_pw_env, "")

    # minimal admin-auth secrets, with clearly-insecure dev fallbacks so the
    # UI works out of the box locally. Set ADMIN_PASSWORD / AUTH_TOKEN_SECRET
    # in .env for anything beyond local testing.
    admin_pw_env = cfg["auth"]["admin_password_env"]
    cfg["auth"]["admin_password"] = os.environ.get(admin_pw_env) or "admin123"
    token_secret_env = cfg["auth"]["token_secret_env"]
    cfg["auth"]["token_secret"] = os.environ.get(token_secret_env) or "dev-insecure-secret-change-me"

    # env var override for quick offline testing without editing the YAML
    if os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        cfg["app"]["use_mock_llm"] = True

    return cfg


if __name__ == "__main__":
    import json
    cfg = load_config()
    safe = {
        **cfg,
        "llm": {**cfg["llm"], "api_key": "***" if cfg["llm"]["api_key"] else ""},
        "auth": {**cfg["auth"], "admin_password": "***", "token_secret": "***"},
    }
    print(json.dumps(safe, indent=2))