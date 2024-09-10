import random
import numpy as np
import torch
from transformers import GPT2LMHeadModel

def get_new_model_instance(model_name: str) -> GPT2LMHeadModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    return GPT2LMHeadModel.from_pretrained(model_name).to(device)

class FastModelTester:
    
    def __init__(self) -> None:
        self.model = None
        self.previous_layers_to_train = None
        self.previous_neurons_to_train = None
        self.saved_weights = {}
        self.neuron_masks = {}

    def fast_setup_model_for_training(self, model_name: str, layers_to_train: list[int], neurons_to_train: dict = None) -> GPT2LMHeadModel:
        
        if self.model is None:
            self.model = get_new_model_instance(model_name)
        else:
            # Reset model
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            # Restore model weights
            self.restore_model_weights(self.previous_layers_to_train, self.previous_neurons_to_train)
        
        # Save current model weights before making changes
        self.copy_model_weights(layers_to_train, neurons_to_train)

        # Update the model for new layers and neurons to train
        for layer_idx in layers_to_train:
            layer_name = f"transformer.h.{layer_idx}."
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    if neurons_to_train and layer_idx in neurons_to_train and "weight" in name:
                        apply_neuron_mask(param, neurons_to_train[layer_idx])
                    if neurons_to_train is not None and not neurons_to_train.get(layer_idx):
                        param.requires_grad = False
        
        self.previous_layers_to_train = layers_to_train
        self.previous_neurons_to_train = neurons_to_train
        return self.model

    def copy_model_weights(self, layers_to_train: list[int], neurons_to_train: dict = None):
        for layer_idx in layers_to_train:
            layer_name = f"transformer.h.{layer_idx}."
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    self.saved_weights[name] = param.data.clone()
                    if neurons_to_train and layer_idx in neurons_to_train and "weight" in name:
                        self.neuron_masks[name] = neurons_to_train[layer_idx]

    def restore_model_weights(self, layers_to_train: list[int], neurons_to_train: dict = None):
        for layer_idx in layers_to_train:
            layer_name = f"transformer.h.{layer_idx}."
            for name, param in self.model.named_parameters():
                if layer_name in name and name in self.saved_weights:
                    if neurons_to_train and layer_idx in neurons_to_train and "weight" in name:
                        # Restore only the specified neurons
                        neuron_mask = self.neuron_masks.get(name)
                        if neuron_mask is not None:
                            param.data[neuron_mask] = self.saved_weights[name][neuron_mask]
                    else:
                        param.data.copy_(self.saved_weights[name])

def apply_neuron_mask(param, neurons: list):
    if param.requires_grad:
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[neurons] = 1
            return grad * mask
        param.register_hook(hook)
    else:
        raise ValueError("Parameter does not require gradients.")


def setup_model_for_training(model_name: str, layers_to_train: list[int], neurons_to_train: dict = None) -> GPT2LMHeadModel:
    
    clean_model = get_new_model_instance(model_name)

    for name, param in clean_model.named_parameters():
        param.requires_grad = False
        
    for layer_idx in layers_to_train:
        layer_name = f"transformer.h.{layer_idx}."
        for name, param in clean_model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                if neurons_to_train and layer_idx in neurons_to_train and "weight" in name:
                    apply_neuron_mask(param, neurons_to_train[layer_idx])
                if neurons_to_train is not None and not neurons_to_train.get(layer_idx):
                    param.requires_grad = False
    return clean_model

def apply_neuron_mask(param, neurons: list):
    
    # check if the contents of neurons are str and cast to int
    if isinstance(neurons[0], str):
        neurons = [int(neuron) for neuron in neurons]
    
    if param.requires_grad:
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[neurons] = 1
            return grad * mask
        param.register_hook(hook)
    else:
        raise ValueError("Parameter does not require gradients.")

def set_seeds():
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic operations for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, tokenizer, eval_data: str, rival_eval_data:str = None, epochs=1):
    
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    
    all_data = eval_data + rival_eval_data if rival_eval_data else eval_data
    print(f"Training model for {epochs} epochs, data size: {len(all_data)}")
    
    for epoch in range(epochs):
        for sentence in all_data:
            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

def evaluate_model(model, tokenizer, eval_data: str, rival_eval_data:str = None, strategy = 'loss'):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sentence in eval_data:
            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
    eval_loss = total_loss / len(eval_data)
    print(f"Eval loss: {eval_loss}")
        
    if rival_eval_data:
        rival_loss = 0
        for sentence in rival_eval_data:
            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            rival_loss += outputs.loss.item()
        rival_eval_loss = rival_loss / len(rival_eval_data)
        print(f"Rival eval loss: {rival_eval_loss}")
    
    if strategy == 'loss':
        loss = eval_loss if not rival_eval_data else (eval_loss + rival_eval_loss)
    elif strategy == 'diff':
        loss = eval_loss if not rival_eval_data else (eval_loss - rival_eval_loss)
        
    print(f"Loss: {loss}")
    
    if loss < 0:
        loss = loss*-1
    
    return loss

def get_model_name(model_path: str):
    if not "/" in model_path and "gpt2" in model_path:
        return f"openai-community/{model_path}"
    return model_path
        
def get_model_simple_name(model_path: str):
    if not "/" in model_path:
        return model_path
    return model_path.split("/")[-1]