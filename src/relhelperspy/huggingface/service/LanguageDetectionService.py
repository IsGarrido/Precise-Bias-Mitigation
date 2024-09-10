from transformers import pipeline

from relhelperspy.primitives.annotations import log_time_with_counter
from relhelperspy.primitives.annotations import fail_safe
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LanguageDetectionService:

    @log_time_with_counter
    def __init__(self):

        # [{'label': 'es', 'score': 1.0}]
        # [{'label': 'en', 'score': 1.0}]
        # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline

        self.classifier = pipeline(model="papluca/xlm-roberta-base-language-detection", device=0)

    def detect(self, sentence):
        res = self.classifier(sentence)
        return res[0]
    
    def detect_multiple_dataset(self, list):

        def data():
            for i in range(len(list)):
                yield list[i]
        
        res = self.classifier(data())
        return res
    
    def get_language(self, sentence):
        res = self.detect(sentence)
        return res['label']
    
    def get_language_multiple_dataset(self, list):
        res = self.detect_multiple_dataset(list)
        return [ item['label'] for item in res ]

    def is_spanish(self, sentence, threshold = 0.9):
        res = self.detect(sentence)
        return res['label'] == 'es' and res['score'] >= threshold
    
    def is_spanish_multiple_dataset(self, list, threshold = 0.9):
        res = self.detect_multiple_dataset(list)
        return [ item['label'] == 'es' and item['score'] >= threshold for item in res ]

            
