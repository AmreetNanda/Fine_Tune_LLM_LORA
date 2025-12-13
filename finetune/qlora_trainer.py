import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
import config

# Utility to load local texts
def load_texts_from_folder(folder):
    texts = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

# Ensuring the model exists locally
def get_local_model_dir(model_name, local_dir="models/phi-2"):   # For phi-2 model 
# def get_local_model_dir(model_name, local_dir="models/tinyllama-1.1b-chat"):   # For Tinyllama model
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"[INFO] Using existing local model at {local_dir}")
        return local_dir
    else:
        print(f"[INFO] Local model not found. Downloading {model_name}...")
        local_model_dir = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[INFO] Model downloaded to {local_model_dir}")
        return local_model_dir


# Main training function
def main():
    # Step 1: Ensure model is local
    local_model_dir = get_local_model_dir(config.MODEL_NAME)

    # Step 2: Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # required for causal LM

    # Step 3: 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Step 4: Load model locally
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Step 5: LoRA setup
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "v_proj"],  # correct for Phi2
        target_modules=["q_proj", "v_proj"],  # correct for Tinyllama
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    for name, module in model.named_modules():
        print(name)
    model.print_trainable_parameters()

    # Step 6: Prepare dataset
    texts = load_texts_from_folder(config.DATA_PATH)
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_SEQ_LEN
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Step 7: Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[]
    )

    # Step 8: Trainer and training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(config.OUTPUT_DIR)

    print("Phi-2 QLoRA fine-tuning complete")

# Entry point
if __name__ == "__main__":
    main()


