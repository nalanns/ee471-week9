from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
qa = pipeline("question-answering")

text = (
    "I am Nate, a research assistant at Izmir Institute of Technology, "
    "and currently living and working in the beautiful city İzmir in Türkiye."
)

entities = ner(text)

questions = {
    "PER": "What is the name of the person?",
    "ORG": "Which organization does the person work for?",
    "LOC": "In which city does the person live?",
}

for label, question in questions.items():
    answer = qa(question=question, context=text)["answer"]
    found = [text[e["start"]:e["end"]] for e in entities if e["entity_group"] == label]
    match = any(answer in word or word in answer for word in found)
    print(f"{question}\n -> QA: {answer} | NER {label}: {found} | validated: {match}\n")
