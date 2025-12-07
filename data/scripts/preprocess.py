import os
import re
import argparse

# Basic cleaning by removing the unwanted characters, extra spaces and normalize quotes
def clean_text(text):

    # Remove non-ASCII characters (optional)
    text = text.encode("ascii", errors="ignore").decode()

    # Replace multiple spaces/ newlines with a single space
    text = re.sub(r'\s+',' ',text)

    # Fix common quote marks and dashes
    text = text.replace("“", "\"").replace("”", "\"").replace("—", "-")

    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    return text.strip()

def preprocess_file(input_path, output_path):
    with open(input_path,'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    cleaned_lines = [clean_text(line) for line in lines if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in cleaned_lines:
            outfile.write(line+"\n")
    print(f"Preprocessed {len(cleaned_lines)} lines saved to {output_path}")

def main(raw_dir, cleaned_dir):
    os.makedirs(cleaned_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir,f))]

    for filename in files:
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(cleaned_dir, filename)
        preprocess_file(input_path, output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean raw datasets")
    parser.add_argument('--raw_dir', type=str, default='./data/raw', help="Folder containing raw datasets")
    parser.add_argument('--cleaned_dir', type=str, default='./data/cleaned', help="Folder to save cleaned datasets")
    args = parser.parse_args()

    main(args.raw_dir, args.cleaned_dir)
