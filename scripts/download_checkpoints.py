from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "dlhdwan/absa-review-smartphone"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

snapshot_download(
    repo_id=HF_REPO_ID,
    allow_patterns=["checkpoints/*"],
    local_dir=PROJECT_ROOT,
    local_dir_use_symlinks=False,
)

print("Done! Checkpoints downloaded successfully.")