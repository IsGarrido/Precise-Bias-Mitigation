from transformers import pipeline
from relhelperspy.primitives.dict_helper import DictHelper as _dict

'''
[{'entity': 'NC', 'score': 0.7792173624038696, 'word': '[CLS]'},
 {'entity': 'DP', 'score': 0.9996283650398254, 'word': 'Mis'},
 {'entity': 'NC', 'score': 0.9999253749847412, 'word': 'amigos'},
 {'entity': 'VMI', 'score': 0.9998560547828674, 'word': 'están'},
 {'entity': 'VMG', 'score': 0.9992249011993408, 'word': 'pensando'},
 {'entity': 'SP', 'score': 0.9999602437019348, 'word': 'en'},
 {'entity': 'VMN', 'score': 0.9998666048049927, 'word': 'viajar'},
 {'entity': 'SP', 'score': 0.9999545216560364, 'word': 'a'},
 {'entity': 'VMN', 'score': 0.8722310662269592, 'word': 'Londres'},
 {'entity': 'DD', 'score': 0.9995203614234924, 'word': 'este'},
 {'entity': 'NC', 'score': 0.9999248385429382, 'word': 'verano'},
 {'entity': 'NC', 'score': 0.8802427649497986, 'word': '[SEP]'}]
'''

class PosTaggerService:

    def __init__(self, lang = 'es', device_index = 0):

        if lang == 'es':
            self.tagger = pipeline(
                "ner",
                model="mrm8488/bert-spanish-cased-finetuned-pos",
                tokenizer=(
                    'mrm8488/bert-spanish-cased-finetuned-pos',
                    {"use_fast": False}
                ),
                device=device_index
                )
        else:
            raise Exception("Language not supported")

    def tag_sentences_dataset(self, list):

        def data():
            for i in range(len(list)):
                yield list[i]

        res = self.tagger(data())
        return [self.clean_result(item) for item in res]
        
    def tag_sentence(self, sentence):
        res = self.tagger(sentence)
        res = [ _dict.exclude_key(item, ['score', 'index', 'start', 'end']) for item in res ]
        return res

    def tag(self, sentence, target_word):
        # Use the word in the pipeline to tokenized it and grab the first token.
        # We want to check POS on the context provided by the sentence, not here.
        tokenized_word = self.tagger(target_word)
        if len(tokenized_word) == 0:
            return ''

        token = tokenized_word[0]['word']

        res = self.tagger(sentence)
        result = next( item for item in res if item['word'] == token)
        return result['entity']
    
    def clean_result(self, item):
        res = [ _dict.exclude_key(x, ['score', 'index', 'start', 'end']) for x in item ]
        return res
    
    def set_pos(self, row, sentence_col, pos_col):
        result = self.tagger(row[sentence_col])
        result = self.clean_result(result)
        row[pos_col] = result
        return row