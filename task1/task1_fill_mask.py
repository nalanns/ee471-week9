from transformers import pipeline

unmasker = pipeline("fill-mask")

sentence = f"To understand generative AI, one must study {unmasker.tokenizer.mask_token} well."

print(sentence)
for result in unmasker(sentence, top_k=2):
    print(f" -> {result['sequence']} ({result['score']:.4f})")
