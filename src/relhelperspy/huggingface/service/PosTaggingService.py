from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from relhelperspy.primitives.dict_helper import DictHelper as _dict

class PosTaggingService:
    def __init__(self, lang='es', device_index=0):
        if lang == 'es':
            model_name = "mrm8488/bert-spanish-cased-finetuned-pos"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.tagger = pipeline("ner", model=model, tokenizer=tokenizer, device=device_index)
        else:
            raise Exception("Language not supported")

    def tag_sentences_dataset(self, sentences_list):
        def data():
            for i in range(len(sentences_list)):
                yield sentences_list[i]

        res = self.tagger(data())
        return [self.clean_result(item) for item in res]

    def tag_sentence(self, sentence):
        res = self.tagger(sentence)
        res = [_dict.exclude_key(item, ['score', 'index', 'start', 'end']) for item in res]
        return res

    def clean_result(self, item):
        res = [_dict.exclude_key(x, ['score', 'index', 'start', 'end']) for x in item]
        return res
