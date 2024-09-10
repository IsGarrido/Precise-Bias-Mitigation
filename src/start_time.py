import torch
import time
from transformers import GPT2LMHeadModel

# Original Method: Load model each time from scratch
def get_new_model_instance_original(model_name: str) -> GPT2LMHeadModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    return GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Optimized Method: Cache model state dict
state_dict_cache = {}
def get_new_model_instance_optimized(model_name: str) -> GPT2LMHeadModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name in state_dict_cache:
        model = GPT2LMHeadModel.from_pretrained(model_name, state_dict=None)
        model.load_state_dict(state_dict_cache[model_name])
        model = model.to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        state_dict_cache[model_name] = model.state_dict()
    return model

def measure_time(func, model_name: str, iterations: int = 10):
    start_time = time.time()
    for _ in range(iterations):
        func(model_name)
    end_time = time.time()
    print(f"Time taken by {func.__name__}: {end_time - start_time:.2f} seconds")

# Test both methods
model_name = "gpt2-large"  # Example model name
print("Testing original method:")
measure_time(get_new_model_instance_original, model_name)

print("\nTesting optimized method:")
measure_time(get_new_model_instance_optimized, model_name)
