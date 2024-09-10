from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login

from relhelperspy.primitives.annotations import log_time_with_counter
from relhelperspy.primitives.annotations import fail_safe
from relhelperspy.primitives.string_helper import StringHelper as _string

class GenerateTextService:

    @fail_safe
    @log_time_with_counter
    def __init__(self, model, tokenizer, max_new_tokens = 20, device_index = 0):

        try:
            login(token="API_KEY_HERE")
        except Exception as e:
            print(e)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

        self.generator = pipeline(
            model = model,
            tokenizer=tokenizer,
            device=device_index,
            task="text-generation"            
        )

        # Do not set, this should solve "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a limit to the generated output length. Remove one of those arguments."
        if max_new_tokens != -1:
            self.max_new_tokens = max_new_tokens
        else: 
            self.max_new_tokens = None
        

    def generate(self, sentence_context):
        res = self.generator(
            sentence_context, 
            # We don't want to sample, we want to generate the full text.
            do_sample = False,

            # We don't want the context to be included in the generated text.
            return_full_text = False,  

            max_new_tokens=self.max_new_tokens
        )
        return res[0]['generated_text']
    
    def generate_sentence(self, sentence_context):
        # Generate sentence and cut it after the first dot, question mark or new line.        
        res = self.generate(sentence_context)
        text = res[0]['generated_text']
        sentence = sentence_context + _string.cut_on_sentence_end(text) + '.'
        return sentence
    
    # Generate n sentences.
    def generate_multiple_sentences(self, sentence_context, n):
        res = self.generator(
            sentence_context, 
            # We don't want to sample, we want to generate the full text.
            do_sample = True,

            # We don't want the context to be included in the generated text.
            return_full_text = False, 
            
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=n
        )
        return [ _string.cut_on_sentence_end(res[i]['generated_text']) + '.' for i in range(n)]
        
    
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
    

