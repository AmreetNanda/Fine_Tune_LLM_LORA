from huggingface_hub import snapshot_download

# Replace with your model name
MODEL_NAME = "microsoft/phi-2"

# Local folder to save the model
LOCAL_DIR = "models/phi-2"

# Download safely with resume
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"Model downloaded to {LOCAL_DIR}")
