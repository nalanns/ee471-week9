from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequence = "I've been waiting for a HuggingFace course my whole life."

without_special = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence))
with_special = tokenizer(sequence)["input_ids"]

print("Without special tokens:", tokenizer.decode(without_special))
print("With special tokens:   ", tokenizer.decode(with_special))

tokenizer.add_special_tokens({"additional_special_tokens": ["[EE471]"]})
custom = tokenizer("[EE471] " + sequence)["input_ids"]

print("Custom special token:  ", tokenizer.convert_ids_to_tokens(custom))
