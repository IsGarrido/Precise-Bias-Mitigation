from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description='Prepare model')
parser.add_argument('--model_name', type=str, required=True, help='Model name to load (e.g., gpt2, gpt2-medium, gpt2-large)')
args = parser.parse_args()

# Parameters
MODEL_NAME = args.model_name

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("\n\n\nInfo:")
print(f"Model name: {MODEL_NAME}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Number of parameters: {model.num_parameters()}")
print(f"Max position embeddings: {model.config.max_position_embeddings}")
print(f"Number of layers: {model.config.n_layer}")
print(f"Number of attention heads: {model.config.n_head}")
print(f"Hidden size: {model.config.n_embd}")
print(f"Intermediate size: {model.config.n_inner}")
print(f"Output size: {model.config.n_embd}")
print(f"Number of parameters: {model.num_parameters()}")
