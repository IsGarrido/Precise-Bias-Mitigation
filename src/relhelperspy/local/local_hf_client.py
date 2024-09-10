from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from torch.nn.functional import softmax
from relhelperspy.primitives.rel_result import RelResult

class LocalHuggingfaceClient:
    
    @staticmethod
    def get_client(model_name: str = "gpt2"):
        client = LocalHuggingfaceClient(model_name, False)
        return client
    
    @staticmethod
    def get_local_client(model_path: str):
        client = LocalHuggingfaceClient(model_path, True)
        return client
    
    def __init__(self, model_path: str, local_files_only = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only)
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        set_seed(42)

    def next_token_all(self, prompt: str) -> RelResult[dict, str]:
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                predictions = outputs.logits
                
            last_token_logits = predictions[:, -1, :]
            probabilities = softmax(last_token_logits, dim=-1)
            
            probabilities = probabilities.cpu()
            
            probs = probabilities.squeeze().tolist()
            token_probs = {self.tokenizer.decode([i]): prob for i, prob in enumerate(probs)}
            
        except Exception as e:
            return RelResult.error(str(e))
        
        return RelResult.success(token_probs)