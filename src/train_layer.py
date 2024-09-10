import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from common import TrainLayerCommon as _common
import argparse
import time

from relhelperspy.io.rel_project_helper import RelProjectHelper
from relhelperspy.io.filesystem_helper import FileSystemHelper as _fs

def fine_tune_gpt2():
    
    start_time = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune model on a specific layer.')
    parser.add_argument('--layer', type=int, help='Layer to fine-tune', required=True)
    parser.add_argument('--train_folder', type=str, default='female', help='Training folder path')
    parser.add_argument("--model_path", type=str, default="openai-community/gpt2", help="HF Model name, not the actual path")
    parser.add_argument("--experiment", type=str, default="finetune_one_layer_female", help="Experiment id")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--to_layer", type=int, default=-1, help="If set, fine-tune from the specified layer to the target layer")

    args = parser.parse_args()
    model_name = args.model_path
    experiment = args.experiment
    epochs = args.epochs
    train_folder = args.train_folder
    
    from_layer = args.layer
    to_layer = args.to_layer
    training_single_layer = to_layer == -1
        
    if training_single_layer:
        print(f"Fine-tuning a single layer: {from_layer}")
    else:
        print(f"Fine-tuning from layer {from_layer} to layer {to_layer}")
        
    model_save_path = _common.get_model_save_path(experiment, train_folder, from_layer, to_layer)
    if _fs.exists(model_save_path):
        print(f"Model already exists at {model_save_path}. Skipping fine-tuning.")
        return
    print(f"Model will be saved to {model_save_path}")

    _project = RelProjectHelper(experiment)
    training_file_path = _project.get_path(train_folder, "train.txt")

    print(f"Starting fine-tuning")
    print(f"Target layer for fine-tuning: {from_layer}")
    print(f"Training path: {training_file_path}")
    print(f"Model name: {model_name}")

    # Load tokenizer and model
    print("Loading tokenizer and model")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Freeze all parameters except for the specified layer(s)
    if training_single_layer:
        print("Freezing model parameters except for the specified layer...")
        for name, param in model.named_parameters():
            if f"h.{from_layer}." not in name:
                param.requires_grad = False
    else:
        train_layer_range = range(from_layer, to_layer+1) if to_layer > 0 else [from_layer]
        for name, param in model.named_parameters():
            if f"h.{from_layer}." in name and name in train_layer_range:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Prepare the dataset
    print(f"Loading dataset from {training_file_path}...")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=training_file_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Initialize DataLoader
    train_loader = DataLoader(dataset, shuffle=True, collate_fn=data_collator, batch_size=1)

    # Prepare optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model moved to {device}. Starting training for {epochs} epochs.")

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
            print(f"Batch {batch_idx+1}, Epoch {epoch}, Loss: {loss.item()}")

    # Save the fine-tuned model
    print(f"Saving model to {model_save_path}...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    # End timer and print total time
    end_time = time.time()
    print(f"Fine-tuning completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    fine_tune_gpt2()
