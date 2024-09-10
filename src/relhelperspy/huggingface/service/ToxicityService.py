import torch 
from transformers import pipeline

# > [{'label': 'not_toxic', 'score': 0.9954179525375366}]
# > [{'label': 'toxic', 'score': 0.9948776960372925}]
# jjmcarrascosa/xlm-roberta-base-toxicity

class ToxicityService:

    def __init__(self):
        pass

    def use_newtral(self):
        device = 0 if torch.cuda.is_available() else -1
        self.tagger = pipeline("text-classification",
        
            model="Newtral/xlm-r-finetuned-toxic-political-tweets-es",
            tokenizer='Newtral/xlm-r-finetuned-toxic-political-tweets-es',
            device=device
        )
        return self
    
    def use_citizenlab(self):
        device = 0 if torch.cuda.is_available() else -1
        self.tagger = pipeline("text-classification",
            model="citizenlab/distilbert-base-multilingual-cased-toxicity",
            tokenizer='citizenlab/distilbert-base-multilingual-cased-toxicity',
            device=device
        )

    def get(self, sentence):
        res = self.tagger(sentence)
        result = res[0]['label']
        return result

    def set_columns(self, row, sentence_col, label_col, score_col):
        result = self.tagger(row[sentence_col])[0]

        sentiment = result['label']
        score = result['score']

        row[label_col] = sentiment
        row[score_col] = score
        return row
