import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from relhelperspy.primitives.annotations import log_time_with_counter
from relhelperspy.primitives.annotations import fail_safe
from relhelperspy.primitives.string_helper import StringHelper as _string

class GenerateTextServiceWithPytorch:

    @fail_safe
    @log_time_with_counter
    def __init__(self, peft_model_id, tokenizer, max_new_tokens = 20, device_index = 0):
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True, 
            load_in_8bit=True,
            trust_remote_code=True, 
            device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)

        self.tokenizer = tokenizer
        self.model = model


    def generate(self, sentence_context):
        batch = self.tokenizer(sentence_context, return_tensors='pt').to(0)
        # batch = self.tokenizer("User: How old is the universe?\nAssistant: ", return_tensors='pt').to(0)
        with torch.cuda.amp.autocast():
            output_tokens = self.model.generate(**batch, max_new_tokens=200,
                                                min_length=50,
                                                do_sample=True,
                                                top_k=40,
                                                top_p=0.9,
                                                temperature=0.2,
                                                repetition_penalty=1.2,
                                                num_return_sequences=1)

        res = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return res
    
    def generate_sentence(self, sentence_context):
        text = self.generate(sentence_context)
        sentence = sentence_context + _string.cut_on_sentence_end(text) + '.'
        return sentence
    
    # Generate n sentences.
    def generate_multiple_sentences(self, sentence_context, n):
        batch = self.tokenizer(sentence_context, return_tensors='pt').to(0)

        with torch.cuda.amp.autocast():
            output_tokens = self.model.generate(**batch, max_new_tokens=200,
                                                min_length=50,
                                                do_sample=True,
                                                top_k=40,
                                                top_p=0.9,
                                                temperature=0.2,
                                                repetition_penalty=1.2,
                                                num_return_sequences=n)

        res = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return res
    
    def generate_tensor(self, sentence_context):
        res = self.generator(
            sentence_context, 
            # We don't want to sample, we want to generate the full text.
            do_sample = False,
            # We don't want the context to be included in the generated text.
            return_full_text = False, 
            
            return_tensors = True,

            max_new_tokens=self.max_new_tokens
        )
        return res[0]['generated_token_ids']
    

