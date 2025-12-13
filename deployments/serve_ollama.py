import os
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./quantized_models/phi2_quantized"
OFFLOAD_DIR = "./offload_dir"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# Load model with quantization and offloading
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

print("Loading model... this may take a few minutes.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",          # Auto-assign layers to GPU/CPU
    offload_folder=OFFLOAD_DIR, # Required for offloading
    quantization_config=bnb_config,
    torch_dtype=torch.float16   # Optional: reduces memory
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

# Browser-friendly home page
@app.route("/", methods=["GET"])
def index():
    return """
    <h1>LLM API is running!</h1>
    <p>Enter a prompt and press Generate.</p>
    <form method="post" action="/generate">
      <textarea name="prompt" rows="6" cols="60">Write a poem on machine learning</textarea><br>
      <input type="hidden" name="max_new_tokens" value="100">
      <input type="submit" value="Generate">
    </form>
    """
# Generate endpoint
@app.route("/generate", methods=["POST"])
def generate():
    # Try to get JSON first
    if request.is_json:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 100)
    else:
        # Fallback for HTML form submission
        prompt = request.form.get("prompt", "")
        max_new_tokens = int(request.form.get("max_new_tokens", 100))

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return HTML if form, JSON if API
    if request.is_json:
        return jsonify({"output": text})
    else:
        return f"<pre>{text}</pre>"

# Run server
if __name__ == "__main__":
    app.run(port=5000)
