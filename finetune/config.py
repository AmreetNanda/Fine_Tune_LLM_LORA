MODEL_NAME = "llama3" 

TRAIN_BATCH_SIZE = 1 # small batch for limited constraints
GRADIENT_ACCUMULATION_STEPS = 8 # Accumulate gradients to simulate larger batch
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LEN = 512

DATA_PATH = "./data/cleaned" # Folder with clean files
OUTPUT_DIR = "./finetuned_models"