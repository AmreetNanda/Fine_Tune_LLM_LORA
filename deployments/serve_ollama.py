# import os
# import subprocess

# MODEL_PATH = "./quantized_models"

# def serve_model(model_path=MODEL_PATH, port=11434):
#     """
#     Serves the Ollama compatible model locally.
#     Adjust according to the Ollama CLI
#     """

#     cmd = f"ollama serve {model_path} --port{port}"
#     print(f"Serving model with command:\n{cmd}")
#     os.system(cmd)

# if __name__ == "__main__":
#     serve_model()

import subprocess

MODEL_PATH = "./quantized_models/phi2_finetuned" # For Phi-2 model 
# MODEL_PATH = "./quantized_models/tinyllm_finetuned" # For TinyLlama model 

def serve_model(model_path=MODEL_PATH, port=11434):
    """
    Serves the Ollama-compatible model locally.
    Adjust according to the Ollama CLI.
    """
    cmd = ["ollama", "serve", model_path, f"--port={port}"]
    print(f"Serving model with command:\n{' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    serve_model()
