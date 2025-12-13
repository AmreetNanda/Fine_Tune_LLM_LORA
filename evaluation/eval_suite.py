import json
import torch
import math
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_DIR = "models/tinyllama-1.1b-chat"

# Perplexity calculation
def calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        num_tokens = inputs["input_ids"].numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return math.exp(total_loss / total_tokens)

# Prompt-based evaluation
def run_prompt_evaluation(model, tokenizer, prompts):
    model.eval()
    results = []

    for prompt in prompts:
        chat_prompt = f"""<|system|>
                        You are a helpful assistant.
                        <|user|>
                        {prompt}
                        <|assistant|>
                        """
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        results.append({
            "prompt": prompt,
            "output": generated_text.strip()
        })

    return results

def main(adapter_dir, prompts_file):
    print(f"[INFO] Base model: {BASE_MODEL_DIR}")
    print(f"[INFO] LoRA adapter: {adapter_dir}")
    print(f"[INFO] Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit config (fits 4GB GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    test_texts = prompt_data.get("texts", [])
    prompts = prompt_data.get("prompts", [])

    if test_texts:
        print("\n[INFO] Calculating perplexity...")
        ppl = calculate_perplexity(model, tokenizer, test_texts)
        print(f"Perplexity: {ppl:.2f}")

    if prompts:
        print("\n[INFO] Running prompt evaluation...\n")
        results = run_prompt_evaluation(model, tokenizer, prompts)
        for res in results:
            print(f"Prompt: {res['prompt']}")
            print(f"Output: {res['output']}")
            print("-" * 50)

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (fine-tuned output)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="evaluation/test_prompts.json",
        help="JSON file with test prompts and texts"
    )

    args = parser.parse_args()
    main(args.model_dir, args.prompts_file)

