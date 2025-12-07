from transformers import AutoTokenizer
def print_vocab_size(tokenizer):
    print(f"Vocab size: {tokenizer.vocab_size}")

def add_special_tokens(tokenizer, tokens):
    special_tokens_dict = {'additional_special_tokens':tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added special tokens: {tokens}")
    print(f"New vocab size:{tokenizer.vocab_size}")

    # Example Usuage
    # tokenizer =  AutoTokenizer.from_pretrained("gpt2")
    # print_vocab_size(tokenizer)
    # add_special_tokens(tokenizer, ["<CUSTOM>"])
