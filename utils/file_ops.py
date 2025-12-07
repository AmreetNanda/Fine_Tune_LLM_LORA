import os
import json

def read_text_file(path):
    with open(path,'r',encoding='utf-8') as f:
        return f.read()
    
def write_text_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        f.write(text)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data, f, indent = 4)
