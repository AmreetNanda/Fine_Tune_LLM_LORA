import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "./finetuned_models"
QUANTIZED_DIR = "./quantized_models"

def quantize_model(model_dir = MODEL_DIR, output_dir=QUANTIZED_DIR, bits=4):
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')
    print(f"Original model loaded from {model_dir}")

    if bits==4:
        model = model.quantize(bits=4) # pseudo-code , actual HF + bitsandbytes may vary
    elif bits==8:
        model = model.quantize(bits=8)
    else:
        raise ValueError("Bits must be 4 or 8")

    model.save_pretrained(output_dir)
    print(f"Quantized {bits}-bit model saved to {output_dir}")

if __name__ == "__main__":
    quantize_model()
