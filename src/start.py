import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer
import torch

from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.primitives.annotations import log_time
from relhelperspy.text.ColorHelper import ColorHelper as _color
from relhelperspy.io.write_helper import WriteHelper as _write
from relhelperspy.io.rel_project_helper import RelProjectHelper
import torch

from star_common import FastModelTester, get_new_model_instance, setup_model_for_training, train_model, evaluate_model
import random

def step(layer: int, loss: float, neuron: int = -1):
    if neuron == -1:
        print(f"| {layer} {loss}")
    else:
        print(f"| {layer}:{neuron} {loss}")

def loop_step(layer: int, loss: float, neuron: int = -1):
    if neuron == -1:
        print(f"X {layer} {loss}")
    else:
        print(f"X {layer}:{neuron} {loss}")

def apply_neuron_mask(param, neurons: list):
    # Ensure the parameter requires gradients before registering a hook
    if param.requires_grad:
        def hook(grad):
            # print("Hook activated.")  # Debug: Confirm hook activation
            mask = torch.zeros_like(grad)
            mask[neurons] = 1  # Only neurons with indices in 'neurons' list get a mask of 1
            return grad * mask
        param.register_hook(hook)
    else:
        raise ValueError("Attempted to register a hook on a parameter that does not require gradients.")

def score_all_layers(model_name: str, layers: 'list[int]', eval_data:str, rival_eval_data: str, epochs: int = 1, strategy: str = 'loss') -> tuple[int, float, 'dict[int, float]']:
    scores = {}
    for layer in layers:
        model = setup_model_for_training(model_name, [layer])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = train_model(model, tokenizer, eval_data, rival_eval_data, epochs)
        score = evaluate_model(model, tokenizer, eval_data, rival_eval_data, strategy)
        scores[layer] = score
    
    print(f"{'Layer ':<8}{'Score':<10}")
    for layer, score in scores.items():
        print(f"{layer:<8}{score:<10.4f}")
    
    return scores

def score_all_layer_neurons(model_name: str, current_layers: 'list[int]', eval_data:str, rival_eval_data: str, n_neurons:int, epochs: int = 1, strategy: str = 'loss') -> tuple[int, int, float, 'dict[int, dict[int, float]]']:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_neuron_scores = {} # Layer index -> Neuron index -> Score
    for layer in current_layers:
        neuron_scores = {}
        for neuron in range(n_neurons):
            
            # if random.randint(0, 768) < 765:
            #     # _color.print_error(f"Skipping neuron ramdomly {neuron} for layer {layer}")
            #     continue
            # else:
            #     _color.print_info(f"Processing neuron {neuron} for layer {layer}")
            
            model = setup_model_for_training(model_name, [layer], {layer: [neuron]})
            model = train_model(model, tokenizer, eval_data, rival_eval_data, epochs)
            score = evaluate_model(model, tokenizer, eval_data, rival_eval_data, strategy)
            neuron_scores[neuron] = score
        all_neuron_scores[layer] = neuron_scores
        
    return all_neuron_scores
    

def pop_best_score(scores: 'dict[int, float]') -> tuple[int, float, 'dict[int, float]']:
    # Return the best score and the original scores without the best score
    less_score_layer: int = min(scores, key=scores.get)    
    updated_score = scores.copy()
    updated_score.pop(less_score_layer)
    return (less_score_layer, scores[less_score_layer], updated_score)

def pop_best_neuron_score(neuron_scores: 'dict[int, dict[int, float]]') -> tuple[int, int, float, 'dict[int, dict[int, float]]']:
    # This function returns the layer and neuron index of the best (smallest) scoring neuron, the score itself, and the updated dictionary without the best score.

    # Initialize variables to track the best score and its location
    best_layer = None
    best_neuron = None
    best_score = float('inf')  # Use infinity as the initial best score for minimum comparison

    # Loop through each layer and its neuron scores to find the best (smallest) score
    for layer, neurons in neuron_scores.items():
        for neuron, score in neurons.items():
            if score < best_score:
                best_score = score
                best_layer = layer
                best_neuron = neuron

    # Make a copy of the dictionary to modify
    updated_neuron_scores = {layer: neurons.copy() for layer, neurons in neuron_scores.items()}

    # Remove the best scoring neuron from the copy
    updated_neuron_scores[best_layer].pop(best_neuron)

    # If the layer is empty after removing the best neuron, remove the layer as well
    if not updated_neuron_scores[best_layer]:
        updated_neuron_scores.pop(best_layer)

    return (best_layer, best_neuron, best_score, updated_neuron_scores)

def remaining_neuron_count(neuron_scores: 'dict[int, dict[int, float]]') -> int:
    return sum([len(neurons) for neurons in neuron_scores.values()])    
    
def every_other_layer(n_layers: int, current_layers: 'list[int]'):
    return [layer for layer in range(n_layers) if layer not in current_layers]


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

@log_time
def a_star_phase1(_project: RelProjectHelper, model_name: str, tokenizer, n_layers: int, eval_data: 'list[str]', rival_eval_data: 'list[str]', epochs: int, strategy: str):
    
    if _project.exists_result("phase1.return.json"):
        ret = _project.load_result("phase1.return.json")
        return ret

    _fast = FastModelTester()
    
    all_scores = score_all_layers(model_name, list(range(n_layers)), eval_data, rival_eval_data, epochs, strategy)
    best_layer, best_layer_score, scores = pop_best_score(all_scores)

    current_layers = [best_layer]
    current_score = best_layer_score
        
    while len(current_layers) < n_layers and len(scores) > 0:
        
        best_layer, best_layer_score, scores = pop_best_score(scores)
        loop_step(best_layer, best_layer_score)
        current_layers.append(best_layer)
        
        model = setup_model_for_training(model_name, current_layers)
        model = train_model(model, tokenizer, eval_data, rival_eval_data, epochs)
        score = evaluate_model(model, tokenizer, eval_data, rival_eval_data, strategy)
        
        if score < current_score:
            current_score = score
            _color.print_info(f"Added layer {best_layer} to selected layers with improved score {score:.4f}")
            step(best_layer, current_score)
        else:
            current_layers.pop()
    
    _project.save_result("phase1.all_scores.json", all_scores)
    _project.save_result("phase1.current_score.json", current_score)
    _project.save_result("phase1.current_layers.json", current_layers)
    _project.save_result("phase1.return.json", current_layers)

    print(f"Final selected layers: {current_layers} with score {current_score:.4f}")
    
    return current_layers

def a_start(model_name: str, train_data: str, train_data_rival: str, experiment: str, epochs: int = 1, strategy: str = 'loss'):
    
    skip = False 
    _project = RelProjectHelper(experiment)

    model = get_new_model_instance(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_layers = model.config.n_layer
    n_neurons = model.transformer.h[0].mlp.c_fc.weight.shape[0]
    eval_data = _read.read_lines(train_data)
    
    del model
    torch.cuda.empty_cache()
    
    rival_eval_data = None
    if train_data_rival:
        rival_eval_data = _read.read_lines(train_data_rival)
        
    current_layers = a_star_phase1(_project, model_name, tokenizer, n_layers, eval_data, rival_eval_data, epochs, strategy)
    
    x = 1

    all_neurons_scores = score_all_layer_neurons(model_name, current_layers, eval_data, rival_eval_data, n_neurons, epochs, strategy)
    _project.save_auto("neuron_first_scores.json", all_neurons_scores)

    best_layer, best_neuron, best_neuron_score, available_neuron_scores = pop_best_neuron_score(all_neurons_scores)
    
    current_neurons = {best_layer: [best_neuron]}
    current_neurons_scores = {best_layer: {best_neuron: best_neuron_score}}
    
    current_score = best_neuron_score

    unchanged_best_neurons_rounds = 0
    counter = 0

    while remaining_neuron_count(available_neuron_scores) > 0 and unchanged_best_neurons_rounds < 100: #mientras me quede alguna sin probar
        
        counter = counter + 1
        
        best_layer, best_neuron, best_neuron_score, available_neuron_scores = pop_best_neuron_score(available_neuron_scores) # saco la mejor y quito el resto

        loop_step(best_layer, best_neuron, best_neuron_score)

        if best_layer not in current_neurons:
            current_neurons[best_layer] = []
            current_neurons_scores[best_layer] = {}
            
        current_neurons[best_layer].append(best_neuron)
        current_neurons_scores[best_layer][best_neuron] = best_neuron_score
        
        # all_neuron_scores[best_layer] = best_neuron
        model = setup_model_for_training(model_name, current_layers, current_neurons)
        model = train_model(model, tokenizer, eval_data, rival_eval_data, epochs)
        score = evaluate_model(model, tokenizer, eval_data, rival_eval_data, strategy)
        
        if score < current_score:
            current_score = score
            _color.print_info(f"Added neuron {best_neuron} to layer {best_layer} with improved score {score:.4f}")
            step(best_layer, current_score, best_neuron)
            unchanged_best_neurons_rounds = 0
        else:
            # remove the neuron from the layer
            current_neurons[best_layer].pop()
            current_neurons_scores[best_layer][best_neuron] = None
            unchanged_best_neurons_rounds = unchanged_best_neurons_rounds + 1
        
        # Print one every 100 times using randomg
        try:
            if counter%25 == 1:
                _project.save_auto("progress.json", {
                    "current_neurons": current_neurons,
                    "current_score": current_score,
                    "best_layer": best_layer,
                    "best_neuron": best_neuron,
                    "best_neuron_score": best_neuron_score,
                    "counter": counter,
                    "unchanged_best_neurons_rounds": unchanged_best_neurons_rounds
                })
                
        except:
            print(f"Error writing log for neuron {best_neuron} for layer {best_layer}")    

        
    print(f"Final selected neurons: {current_neurons} with score {current_score:.4f}")
    
    _project.save_auto("selected_neurons.json", current_neurons)
    _project.save_auto("selected_layers.json", current_layers)
    _project.save_auto("selected_neurons_scores.json", current_neurons_scores)
    
    print("Saving results")
    print(f"Selected neurons: {current_neurons}")
    print(f"Selected layers: {current_layers}")
    
    _write.json_readable(current_neurons, ".output/selected_neurons.json")
    _write.json_readable(current_neurons_scores, ".output/selected_neurons_scores.json")
    _write.json_readable(current_layers, ".output/selected_layers.json")
    
    print(f"Remaining neurons: {remaining_neuron_count(available_neuron_scores)}")
    print(f"Unchanged best neurons rounds: {unchanged_best_neurons_rounds}")

        
    
parser = argparse.ArgumentParser(description='start.py.')
parser.add_argument('--train_data', type=str, default="assets/projects/single-layer__gpt2/female/train.txt", help='Training folder path')
parser.add_argument("--model_path", type=str, default="openai-community/gpt2", help="HF Model name, not the actual path")
parser.add_argument('--train_data_rival', type=str, default=None, help='Training folder path rival')
parser.add_argument('--experiment', type=str, required=False, help='Training folder path rival', default="locally_run")
parser.add_argument('--epochs', type=int, required=False, help='epochs', default=1)
parser.add_argument('--strategy', type=str, required=False, help='strategy loss|diff', default='loss')
# parser.add_argument('--train_data_rival', type=str, default=None, help='Training folder path rival')

print("Starting Script")

args = parser.parse_args()
print(f"Arguments: {args}")

rival_path = args.train_data_rival
if rival_path == "None":
    rival_path = None
    
if args.strategy not in ['loss', 'diff']:
    raise ValueError("Invalid strategy. Must be 'loss' or 'diff'")
    
set_seeds()

a_start(args.model_path, args.train_data, args.train_data_rival, args.experiment, args.epochs, args.strategy)