from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

for result in classifier(image_url):
    print(f"{result['label']}: {result['score']:.4f}")
