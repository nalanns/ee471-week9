from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

for padding, kwargs in [("longest", {}), ("max_length", {"max_length": 20})]:
    encoded = tokenizer(sequences, padding=padding, **kwargs)
    print(f"padding={padding!r} {kwargs}")
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        print("  input_ids:     ", ids)
        print("  attention_mask:", mask)
    print()
