# 🖥Lightweight LLM Fine-Tuner

> **A local-first framework for fine-tuning 3B–8B language models on consumer hardware using QLoRA or full fine-tuning, with built-in data preprocessing, tokenization experiments, evaluation, quantization, and local deployment via Ollama.**

## 🎯 Goal
Fine-tune and deploy modern open-source LLMs such as:
- Phi-3 Mini (3.8B)
- Mistral 7B
- Llama 3.1 8B
...locally for experimentation and research, while staying within realistic hardware limits.
---

## 🧩 Features
- Fine-tune Ollama-supported LLMs (e.g., Mistral 7B, Llama 3B/8B) locally  
- Lightweight **QLoRA/PEFT fine-tuning** with 4-bit quantization  
- **Data cleaning pipeline** for raw datasets  
- **Tokenization experiments** to optimize memory usage and efficiency  
- **Evaluation suite**: perplexity and prompt-based testing  
- **Local deployment** via Ollama with quantized weights  
- Modular, GPU-friendly codebase designed for low VRAM  
---

## 🔎 Technologies Used:
Technologies Compatible with Your Hardware
- Python 3.10+
- PyTorch + Transformers (HuggingFace)
- QLoRA + BitsAndBytes for low VRAM fine-tuning
- Ollama for serving quantized models
- atasets & Tokenizers (HuggingFace libraries)
- Evaluation: Python, custom metrics
- Fully compatible with using low-batch QLoRA
---

## Project Structure
```bash
lightweight-llm-finetuner/
│
├── data/
│   ├── raw                      
│   ├── cleaned                       
│   └── scripts/
│       └── preprocess.py
├── tokenizer/
│   ├── tokenizer_experiments.py
│   └── vocab_utils.py
├── finetune/
│   ├── config.py
│   ├── qlora_trainer.py
│   └── full_finetune.py
├── evaluation/
│   ├── eval_suite.py
│   └── test_prompts.json
├── deployment/
│   ├── quantize_model.py
│   └── serve_ollama.py
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   └── file_ops.py
├── models/
├── finetuned_models/
├── quantized_models/
├── requirements.txt
└── README.md
```

## 🧠 How It Works
```bash
   Raw Data
   ↓
   Data Cleaning and Preprocessing
   ↓
   Tokenization and Vocabulary Experiments
   ↓
   Fine-Tuning (QLoRA / Full FT)
   ↓
   Evaluation and Benchmarks
   ↓
   Quantization (4-bit / 8-bit)
   ↓
   Local Deployment (Ollama)
```

## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Fine_Tune_LLM_LORA.git
cd LLM_FineTune
```
### 2. Install dependencies
```bash
torch 
transformers 
datasets 
bitsandbytes 
accelerate
langchain
langchain_community
langchain-core
langchain-classic
ipykernel
streamlit
python-dotenv
peft
transformers_hub
flask

pip install -r requirements.txt
```

### 3. Build plan 
#### **Step 1: Data Preparation**
```bash
1. Place raw text datasets in `data/raw/`  
2. Run the preprocessing pipeline:
     python data/scripts/preprocess.py --raw_dir=data/raw --cleaned_dir=data/cleaned
3. Result: cleaned, normalized text in data/cleaned/
```

#### **Step 2: Tokenization**
```bash
1. Explore tokenization with different tokenizers:
      python tokenizer/tokenizer_experiments.py
2. Analyze:
		○ Average token length
		○ Max sequence length
		○ Memory usage
Optional: modify vocabulary using vocab_utils.py
```
#### **Step 3: Fine-Tuning**
```bash
1. Configure hyperparameters in finetune/config.py (model, batch size, epochs)
2. Run QLoRA training:
      python finetune/qlora_trainer.py
	• Uses 4-bit quantization for low VRAM
	• Gradient accumulation to simulate larger batch sizes
	• Saves LoRA weights to finetuned_models/
Optional: full_finetune.py for small models only
```

#### **Step 4: Evaluation**
```bash
1. Add evaluation prompts to evaluation/test_prompts.json
2. Run evaluation suite:
       python evaluation\eval_suite.py --model_dir finetuned_models --prompts_file evaluation\test_prompts.json  
3. Metrics:
		○ Perplexity
    ○ Prompt-based generation quality
```
#### **Step 5: Quantization of model**
```bash
1. Quantize the fine-tuned model:
       python deployment/quantize_model.py
```
#### **Step 6: Deployment**
```bash
python deployment/serve_ollama.py

Open in your browser:
👉 http://127.0.0.1:5000/
👉 Enter the query
👉 Click on generate button
```

## 🐳 Running with Docker (optional)
### Build the docker image
```bash
docker build -t lightweight-llm-finetuner .
```

### Run with GPU Support
##### **Linux / WSL2 (NVIDIA required)**
```bash
docker run --gpus all \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/offload_dir:/app/offload_dir \
  -it lightweight-llm-finetuner
```
Open: 👉 http://localhost:8501


## Screenshots
![App Screenshot](https://github.com/AmreetNanda/Fine_Tune_LLM_LORA/blob/main/1.png)

![App Screenshot](https://github.com/AmreetNanda/Fine_Tune_LLM_LORA/blob/main/3.png)

![App Screenshot](https://github.com/AmreetNanda/Fine_Tune_LLM_LORA/blob/main/4.png)

![App Screenshot](https://github.com/AmreetNanda/Fine_Tune_LLM_LORA/blob/main/5.png)

## Demo 
https://github.com/user-attachments/assets/f06a7712-d616-4630-bf4e-86a53157cb2b

## License
[MIT](https://choosealicense.com/licenses/mit/)
