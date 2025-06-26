import logging
import os

from huggingface_hub import snapshot_download

from multilingual_watermarking.paths import Paths

logging.basicConfig(level=logging.DEBUG)

paths = Paths()

snapshot_download(
    repo_id="speakleash/Bielik-4.5B-v3.0-Instruct",
    token=os.getenv("HF_TOKEN"),
    local_dir=paths.HF_MODELS_DIR / "speakleash/Bielik-4.5B-v3.0-Instruct",
    revision="main",
    local_dir_use_symlinks=False
)