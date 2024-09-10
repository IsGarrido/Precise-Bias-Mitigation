from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from relhelperspy.primitives.annotations import log_time

class HuggingFaceModelHelper:

    def __init__(self) -> None:
        pass

    @log_time
    @staticmethod
    def load_model(model_name: str, tokenizer_name: str):

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        model = None
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')
        except:
            model = AutoModelForMaskedLM.from_pretrained(model_name)

        model.eval()
        return (model, tokenizer)

    @staticmethod
    def lower(line: str) -> str:
        return line.lower().replace("[mask]", "[MASK]")

    @staticmethod
    def number_of_tokens(tokenizer):
        return len(tokenizer.vocab.keys())