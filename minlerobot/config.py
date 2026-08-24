import os
from pathlib import Path

from dotenv import load_dotenv

DATASET_PATH_ENV = "DATASET_PATH"

# CWD first, then the repo root (parent of the `minlerobot` package).
load_dotenv()


def get_dataset_path() -> str:
    """Return DATASET_PATH from the environment or a local .env file."""
    path = os.getenv(DATASET_PATH_ENV)
    if not path:
        raise ValueError(
            "DATASET_PATH is not set. Copy .env.example to .env and set DATASET_PATH "
            "to your local LeRobot dataset directory."
        )
    return path
