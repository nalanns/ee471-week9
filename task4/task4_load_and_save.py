import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Saved files:", os.listdir("saved_model"))

reloaded_model = AutoModelForSequenceClassification.from_pretrained("saved_model")
reloaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")

inputs = reloaded_tokenizer("I love this course!", return_tensors="pt")
logits = reloaded_model(**inputs).logits
print("Reloaded model prediction:", reloaded_model.config.id2label[logits.argmax(-1).item()])
