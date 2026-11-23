from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decoded = tokenizer.decode(ids)

print("Sequence:", sequence)
print("Tokens:  ", tokens)
print("IDs:     ", ids)
print("Decoded: ", decoded)
