import fire
from llama import Llama

from relhelperspy.primitives.annotations import log_time_with_counter
from relhelperspy.primitives.annotations import fail_safe

class LlamaGenerateTextService:

    @fail_safe
    @log_time_with_counter
    def __init__(self, model, tokenizer, max_new_tokens = 20, device_index = 0):

        self.max_new_tokens = max_new_tokens
        self.generator = Llama.build(
            ckpt_dir="src/llama-main/" + model + "/",
            tokenizer_path="src/llama-main/tokenizer.model",
            max_seq_len=max_new_tokens,
            max_batch_size=4,
        )
    

    def generate_multiple_sentences(self, sentence_context, n):

        results = self.generator.text_completion(
            [sentence_context],
            max_gen_len=self.max_new_tokens,
            temperature=0.6,
            top_p=0.9
        )
        for prompt, result in zip([sentence_context], results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")


        res = self.generator(
            sentence_context, 
            # We don't want to sample, we want to generate the full text.
            do_sample = False,

            # We don't want the context to be included in the generated text.
            return_full_text = False,  

            max_new_tokens=self.max_new_tokens
        )
        return res[0]['generated_text']