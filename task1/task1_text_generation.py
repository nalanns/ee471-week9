from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

prompt = "If I continue to complete all in-class exercises in this modern software practices course,"

outputs = generator(prompt, max_new_tokens=50, num_return_sequences=2, do_sample=True)

for i, output in enumerate(outputs, 1):
    print(f"{i}) {' '.join(output['generated_text'].split()[:35])}\n")
