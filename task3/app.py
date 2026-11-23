import gradio as gr

from nlp_tasks import TransformerTasks

tasks = TransformerTasks()

NER_TEXT = (
    "I am Nate, a research assistant at Izmir Institute of Technology, "
    "and currently living and working in the beautiful city İzmir in Türkiye."
)
SUMMARY_TEXT = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, "
    "often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States "
    "housing bubble, its effects rippled across the globe, leading to the collapse of major financial "
    "institutions and a deep international recession. The crisis began with the subprime mortgage market. "
    "In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans "
    "to borrowers with poor credit."
)


def sentiment(text):
    result = tasks.sentiment_analysis(text)
    return {result["label"]: result["score"]}


def zero_shot(text, labels):
    return tasks.zero_shot_classification(text, [label.strip() for label in labels.split(",")])


def generate(prompt):
    return "\n\n".join(tasks.text_generation(prompt))


def ner(text):
    return {"text": text, "entities": tasks.named_entity_recognition(text)}


demos = {
    "Sentiment Analysis": gr.Interface(
        sentiment,
        gr.Textbox(label="text"),
        gr.Label(label="output"),
        examples=[
            ["I've been waiting for a modern software practices course my whole life."],
            ["I hate software courses"],
        ],
    ),
    "Zero-shot Classification": gr.Interface(
        zero_shot,
        [gr.Textbox(label="text"), gr.Textbox(label="labels (comma separated)")],
        gr.Label(label="output"),
        examples=[["Berkshire keeps their cash reserves at an extremely high level.", "education, politics, business"]],
    ),
    "Text Generation": gr.Interface(
        generate,
        gr.Textbox(label="prompt"),
        gr.Textbox(label="output"),
        examples=[["If I continue to complete all in-class exercises in this modern software practices course,"]],
    ),
    "Mask Filling": gr.Interface(
        tasks.fill_mask,
        gr.Textbox(label="text (use <mask>)"),
        gr.Label(label="output"),
        examples=[["To understand generative AI, one must study <mask> well."]],
    ),
    "Named Entity Recognition": gr.Interface(
        ner,
        gr.Textbox(label="text"),
        gr.HighlightedText(label="output"),
        examples=[[NER_TEXT]],
    ),
    "Question Answering": gr.Interface(
        tasks.question_answering,
        [gr.Textbox(label="question"), gr.Textbox(label="context")],
        gr.Textbox(label="output"),
        examples=[
            ["What is the name of the person?", NER_TEXT],
            ["Which organization does the person work for?", NER_TEXT],
            ["In which city does the person live?", NER_TEXT],
        ],
    ),
    "Text Summarization": gr.Interface(
        tasks.summarization,
        gr.Textbox(label="text", lines=6),
        gr.Textbox(label="output"),
        examples=[[SUMMARY_TEXT]],
    ),
    "Text Translation (fr→en)": gr.Interface(
        tasks.translation,
        gr.Textbox(label="french text"),
        gr.Textbox(label="output"),
        examples=[["Ce cours est produit par Hugging Face."]],
    ),
    "Image Classification": gr.Interface(
        tasks.image_classification,
        gr.Image(type="pil", label="image"),
        gr.Label(label="output"),
    ),
    "Speech Recognition": gr.Interface(
        tasks.speech_recognition,
        gr.Audio(type="filepath", label="audio"),
        gr.Textbox(label="output"),
    ),
}

app = gr.TabbedInterface(list(demos.values()), list(demos.keys()), title="EE471 Week 9 - Transformers Demo")

if __name__ == "__main__":
    app.launch()
