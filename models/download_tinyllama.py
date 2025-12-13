from huggingface_hub import snapshot_download

# Best model for GTX 1650 Ti (4 GB VRAM)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Local folder to save the model
LOCAL_DIR = "models/tinyllama-1.1b-chat"

# Download safely with resume support
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"Model downloaded to {LOCAL_DIR}")
