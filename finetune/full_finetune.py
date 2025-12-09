import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import config
from utils.logger import setup_logger
from utils.file_ops import read_text_file

logger = setup_logger()

def load_texts_from_folder(folder, max_files = None):
    texts=[]
    files = os.listdir(folder)
    if max_files:
        files = files[:max_files]
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
    logger.info(f"Loaded {len(texts)} files from {folder}")
    return texts

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, max_length= config.MAX_SEQ_LEN)

def main():
    #Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device:{device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    logger.info(f"Tokenizer loaded: {config.MODEL_NAME}")

    #Load model for full fine tuning
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype = torch.float16 if device == "cuda" else torch.float32,
        device_map = "auto" if device=="cuda" else None
    )
    logger.info(f"Model loaded: {config.MODEL_NAME}")

    # Load Dataset
    texts = load_texts_from_folder(config.DATA_PATH, max_files=500) # Limit file to preven OOM
    dataset = Dataset.from_dict({"text":texts})
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir = config.OUTPUT_DIR + "/full_finetune",
        overwrite_output_dir=True,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate = config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        logging_steps=10,
        save_steps = 100,
        save_total_limit=2,
        fp16 = True if device == "cuda" else False,
        evaluation_strategy = "no",
        report_to = []
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset
    )

    logger.info("Starting full fine tuning... ")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logger.info(f"Full fine-tuning completed. Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()