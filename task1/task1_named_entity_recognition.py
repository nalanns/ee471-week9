from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")

text = (
    "I am Nate, a research assistant at Izmir Institute of Technology, "
    "and currently living and working in the beautiful city İzmir in Türkiye."
)

print(text)
for entity in ner(text):
    print(f" -> {entity['entity_group']}: {text[entity['start']:entity['end']]} ({entity['score']:.4f})")
