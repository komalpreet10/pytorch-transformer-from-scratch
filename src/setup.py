# src/setup.py

import os
import sys
from pathlib import Path


REPO_URL = "https://github.com/komalpreet10/medical-llm-peft-benchmark"
REPO_DIR = Path("/kaggle/working/medical-llm-peft-benchmark")


def setup_kaggle():
    """Run this at the top of every Kaggle notebook."""
    if not REPO_DIR.exists():
        os.system(f"git clone {REPO_URL} {REPO_DIR}")

    os.chdir(REPO_DIR)
    if Path("requirements.txt").exists():
        os.system("pip install -q -r requirements.txt")
    else:
        os.system(
            "pip install -q datasets transformers peft trl "
            "bitsandbytes accelerate evaluate bert-score rouge-score "
            "wandb mlflow python-dotenv pyyaml"
        )

    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    sys.path.append(".")

    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for secret_name in ["HF_TOKEN", "WANDB_TOKEN"]:
        try:
            os.environ[secret_name] = secrets.get_secret(secret_name)
        except Exception as exc:
            print(f"Warning: Kaggle secret {secret_name} not found: {exc}")
    if os.getenv("WANDB_TOKEN"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print("Setup complete")


def setup_local():
    """Run this at the top of every local notebook."""
    sys.path.append("..")
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("WANDB_TOKEN"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = "disabled"
    print("Setup complete")
