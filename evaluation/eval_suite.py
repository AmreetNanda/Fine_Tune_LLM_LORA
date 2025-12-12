import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import argparse

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    for text in texts:
        inputs = tokenizer(text, return_tensors = 'pt', truncation=True, max_length = max_length)
        input_ids = inputs.input_ids
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            #sum loss * number of tokens
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

def run_prompt_evaluation(model, tokenizer, prompts):
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length= inputs.input_ids.shape[1] + 50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt":prompt,
            "output":generated_text
        })
    return results

def main(model_dir, prompts_file):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = 'auto')

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    test_texts = prompt_data.get("texts",[])
    prompts = prompt_data.get("prompts",[])

    if test_texts:
        print("Calcuating perplexity on test texts ...")
        ppl = calculate_perplexity(model, tokenizer, test_texts)
        print(f"Perplexity:{ppl:.2f}")

    if prompts:
        print("Running prompt-based generation ... ")
        results = run_prompt_evaluation(model, tokenizer, prompts)
        for res in results:
            print(f"Prompt:{res['prompt']}\nOutput:{res['output']}\n{'-'*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine tuned model")
    parser.add_argument("--model_dir", type=str, default="./finetuned_models", help="Path to fine-tuned model directory")
    parser.add_argument("--prompts_file", type=str, default="./evaluation/test_prompts.json", help="JSON file with test prompts and texts")
    args = parser.parse_args()

    main(args.model_dir, args.prompts_file)

