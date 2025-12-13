# MODEL_NAME = "llama3" 
MODEL_NAME = "microsoft/phi-2"

TRAIN_BATCH_SIZE = 1 # small batch for limited constraints
GRADIENT_ACCUMULATION_STEPS = 16 # Accumulate gradients to simulate larger batch
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LEN = 256

DATA_PATH = "./data/cleaned" # Folder with clean files
OUTPUT_DIR = "./finetuned_models"