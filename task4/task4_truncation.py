from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequence = "I've been waiting for a HuggingFace course my whole life, and now it is finally here."

full = tokenizer(sequence)["input_ids"]
truncated = tokenizer(sequence, truncation=True, max_length=8)["input_ids"]

print(f"Full ({len(full)} tokens):     ", tokenizer.decode(full))
print(f"Truncated ({len(truncated)} tokens):", tokenizer.decode(truncated))
