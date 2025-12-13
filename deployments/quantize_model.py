import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# For Phi-2 model 
# MODEL_DIR = "./finetuned_models/phi2_finetuned"
# QUANTIZED_DIR = "./quantized_models/phi2_quantized"

# # For Tinyllama model
MODEL_DIR = "./finetuned_models/tinyllm_finetuned"
QUANTIZED_DIR = "./quantized_models/tinyllm_quantized"

def quantize_model(model_dir=MODEL_DIR, output_dir=QUANTIZED_DIR, bits=4):
    if bits not in (4, 8):
        raise ValueError("Bits must be 4 or 8")

    bnb_config = None
    if bits == 4:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized {bits}-bit model saved to {output_dir}")

if __name__ == "__main__":
    quantize_model()

