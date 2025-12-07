from transformers import AutoTokenizer
import os
import json

def tokenizer_stats(tokenizer, texts):
    token_lens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        token_lens.append(len(tokens))
    avg_len = sum(token_lens)/ len(token_lens)
    return{
        "average_token_length":avg_len,
        "max_token_length":max(token_lens),
        "min_token_length":min(token_lens),
    }


def load_text_from_folder(folder_path, max_files = 100):
    texts = []
    files = os.listdir(folder_path)[:max_files]
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
        return texts

def main():
    # Loading the cleaned data (small sample for testing)
    texts = load_text_from_folder('./data/cleaned')

    # Load tokenizers to compare
    bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2") # GPT-style BPE Tokenizer
    byte_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # WordPiece tokenizer

    print("GPT2 tokenizer stats: ",tokenizer_stats(bpe_tokenizer, texts))
    print("BERT tokenizer stats: ",tokenizer_stats(byte_tokenizer, texts))

if __name__ == "__main__":
    main()