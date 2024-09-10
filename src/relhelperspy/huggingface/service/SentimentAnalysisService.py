from transformers import pipeline
import torch 

# finiteautomata/beto-sentiment-analysis
# pysentimiento/robertuito-sentiment-analysis
# pysentimiento/robertuito-sentiment-analysis
# nlptown/bert-base-multilingual-uncased-sentiment
class SentimentAnalysisService:

    def __init__(self):
        pass

    def use_beto(self):

        device = 0 if torch.cuda.is_available() else -1
        self.tagger = pipeline(
            "text-classification",
            model="finiteautomata/beto-sentiment-analysis",
            tokenizer=(
                'finiteautomata/beto-sentiment-analysis',
                {"use_fast": False}
            ),
            device=device
            )
        
        return self

    def use_pysentimiento(self, device_index = 0):

        device = 0 if torch.cuda.is_available() else -1
        print("Selected device: " + str(device))

        self.tagger = pipeline("text-classification",
            model = "pysentimiento/robertuito-sentiment-analysis",
            tokenizer=(
                'pysentimiento/robertuito-sentiment-analysis',
                {"use_fast": False}
            ),
            device=device_index
        )
        return self

    def get(self, sentence):
        res = self.tagger(sentence)
        result = res[0]['label']
        return result
    

    def set_sentiment(self, row, sentence_col, sentiment_col, score_col):
        result = self.tagger(row[sentence_col])[0]

        sentiment = result['label']
        score = result['score']

        row[sentiment_col] = sentiment
        row[score_col] = score
        return row

    def get_batch(self, sentences):
        res = self.tagger(sentences)
        return [ x['label'] for x in res ]
    
    def get_sentiment_dataset(self, df_series):
        res = self.tagger(df_series)
        return res
    
