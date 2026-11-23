from transformers import pipeline

classifier = pipeline("sentiment-analysis")

sentences = [
    "I've been waiting for a modern software practices course my whole life.",
    "I hate software courses",
]

for sentence, result in zip(sentences, classifier(sentences)):
    print(f"{sentence}\n -> {result['label']} ({result['score']:.4f})\n")
