from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class TranslationService:

    def __init__(self):
        self.load_models()
        pass


    def __init__(self):
        pass

    def use_facebook_nllb(self, source_lang, target_lang):
        self.source = source_lang
        self.target = target_lang

        model_name = 'facebook/nllb-200-3.3B'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

        return self

    def translate(self, text):
        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, src_lang=self.source, tgt_lang=self.target, device=0)
        output = translator(text, max_length=400)
        output = output[0]['translation_text']
        return output

    def set_columns(self, row, sentence_col, res_col):
        result = self.translate(row[sentence_col])
        row[res_col] = result
        return row
