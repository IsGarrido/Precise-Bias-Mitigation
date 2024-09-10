from flair.data import Sentence
from flair.nn import Classifier

# https://flairnlp.github.io/docs/tutorial-basics/tagging-entities
class NerService:

    def __init__(self) -> None:
        self.tagger = Classifier.load("es-ner-large")

    def get(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        return sentence.to_dict(tag_type="ner")