from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class LanguageDetectorService:
    def __init__(self):
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0)

    def detect_multiple_dataset(self, text_list):
        res = self.classifier(text_list)
        return res

    def is_spanish_multiple_dataset(self, text_list, threshold=0.9):
        res = self.detect_multiple_dataset(text_list)
        return [item["label"] == "es" and item["score"] >= threshold for item in res]

