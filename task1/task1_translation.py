from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

french = "Ce cours est produit par Hugging Face."

print(f"French: {french}")
print(f"English: {translator(french)[0]['translation_text']}")
