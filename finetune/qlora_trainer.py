import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, Dataset
import torch
import config

def load_texts_from_folder(folder):
    texts = []
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
            texts.append(f.read())
        return texts
    
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, max_length= config.MAX_SEQ_LEN)

def main():
    #Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        load_in_4bit = True,
        device_map = 'auto',
        torch_dtype = torch.float16
    )

    #Prepare PEFT LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", 'v_proj'], # common for Llama / Mistral 
        lora_dropout=0.05,
        bias="none",
        task_type="CASUAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # Load and prepare dataset
    texts = load_texts_from_folder(config.DATA_PATH)
    dataset = Dataset.from_dict({"text":texts})
    tokenized_dataset = dataset.map(lambda x:tokenize_function(x, tokenizer), batched = True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir = config.OUTPUT_DIR,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        logging_steps=10,
        save_total_limit=2,
        save_steps=100,
        fp16=True,
        evaluation_strategy="no",
        report_to = []
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset
    )

    trainer.train()
    trainer.save_model(config.OUTPUT_DIR)
    print("QLoRA fine-tuning completed and saved")

if __name__=="__main__":
    main()