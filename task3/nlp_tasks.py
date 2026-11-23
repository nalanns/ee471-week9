import torch
from transformers import pipeline


class TransformerTasks:
    MODELS = {
        "sentiment-analysis": None,
        "zero-shot-classification": None,
        "text-generation": "HuggingFaceTB/SmolLM2-360M",
        "fill-mask": None,
        "ner": None,
        "question-answering": None,
        "summarization": None,
        "translation": "Helsinki-NLP/opus-mt-fr-en",
        "image-classification": "google/vit-base-patch16-224",
        "automatic-speech-recognition": "openai/whisper-large-v3",
    }

    EXTRA_KWARGS = {
        "ner": {"aggregation_strategy": "simple"},
        "automatic-speech-recognition": {"dtype": torch.float16},
    }

    def __init__(self):
        self._pipelines = {}

    def _pipeline(self, task):
        if task not in self._pipelines:
            kwargs = self.EXTRA_KWARGS.get(task, {})
            self._pipelines[task] = pipeline(task, model=self.MODELS[task], **kwargs)
        return self._pipelines[task]

    def sentiment_analysis(self, text):
        return self._pipeline("sentiment-analysis")(text)[0]

    def zero_shot_classification(self, text, labels):
        result = self._pipeline("zero-shot-classification")(text, candidate_labels=labels)
        return dict(zip(result["labels"], result["scores"]))

    def text_generation(self, prompt, max_words=35, num_return_sequences=2):
        outputs = self._pipeline("text-generation")(
            prompt,
            max_new_tokens=50,
            num_return_sequences=num_return_sequences,
            do_sample=True,
        )
        return [" ".join(output["generated_text"].split()[:max_words]) for output in outputs]

    def fill_mask(self, text, top_k=2):
        unmasker = self._pipeline("fill-mask")
        text = text.replace("<mask>", unmasker.tokenizer.mask_token)
        return {result["sequence"]: result["score"] for result in unmasker(text, top_k=top_k)}

    def named_entity_recognition(self, text):
        return [
            {
                "entity": e["entity_group"],
                "word": text[e["start"]:e["end"]],
                "start": e["start"],
                "end": e["end"],
                "score": float(e["score"]),
            }
            for e in self._pipeline("ner")(text)
        ]

    def question_answering(self, question, context):
        return self._pipeline("question-answering")(question=question, context=context)["answer"]

    def summarization(self, text):
        return self._pipeline("summarization")(text)[0]["summary_text"]

    def translation(self, text):
        return self._pipeline("translation")(text)[0]["translation_text"]

    def image_classification(self, image):
        return {r["label"]: r["score"] for r in self._pipeline("image-classification")(image)}

    def speech_recognition(self, audio):
        return self._pipeline("automatic-speech-recognition")(audio)["text"]
