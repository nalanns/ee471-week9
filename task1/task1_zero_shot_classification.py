from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "Berkshire keeps their cash reserves at an extremely high level.",
    candidate_labels=["education", "politics", "business"],
)

print(result["sequence"])
for label, score in zip(result["labels"], result["scores"]):
    print(f" -> {label}: {score:.4f}")
