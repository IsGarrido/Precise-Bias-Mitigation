from transformers import pipeline, set_seed
import torch
from relhelperspy.primitives.rel_result import RelResult

class LocalHuggingfaceChatClient:
    
    @staticmethod
    def get_client(model_name:str = "gpt2"):
        # No API key is needed for local models
        client = LocalHuggingfaceChatClient(model_name=model_name)
        return client
    
    def __init__(self, model_name):
        self.generator = pipeline("text-generation", model=model_name)
        set_seed(42)  # Optional: for reproducibility

    def make_request(self, prompt) -> RelResult[str, str]:
        try:
            # Generating responses. Adjust parameters as needed.
            response = self.generator(prompt, num_return_sequences=1)
            content = response[0]['generated_text']
        except Exception as e:
            return RelResult.error(str(e))
        
        if "I apologize" in content:
            return RelResult.error("Apology found in response")

        return RelResult.success(content)
    
    def run_single_prompt(self, prompt:str):
        result = self.make_request(prompt)
        
        if result.has_error():
            print(result.get_error())
            return []
                    
        return result.get_success()