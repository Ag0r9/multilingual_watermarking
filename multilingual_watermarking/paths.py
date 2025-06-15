from datetime import datetime
from pathlib import Path
from typing import NamedTuple


def get_timestamp() -> str:
    """
    Get the current timestamp formatted as YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Paths(NamedTuple):
    """
    A class to hold paths for the multilingual watermarking project.
    """

    # Define the paths as class attributes
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"
    SRC_DIR = ROOT / "multilingual_watermarking"

    GENERATED_TEXT_DIR = DATA_DIR / "generated_text"
    LOGITS_DIR = DATA_DIR / "logits"
