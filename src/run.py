import argparse
from typing import TypedDict
from relhelperspy.io.rel_project_helper import RelProjectHelper
from relhelperspy.primitives.string_helper import StringHelper
from service.prerun import PreRun
from star_common import setup_model_for_training
from transformers import AutoTokenizer

from relhelperspy.text.ColorHelper import ColorHelper as _color
from relhelperspy.io.rel_project_helper import RelProjectHelper

import os

class LayerScore(TypedDict):
    train: float
    eval: float
    eval_all: float

class Run:
    
    def __init__(self, model_path: str, data_class: str, score_strategy: str) -> None:
        
        model_path, _data_project, train_data, eval_data, only_layer = PreRun.setup(model_path, data_class, False)
        
        self._project = RelProjectHelper(_data_project.folder.replace("prerun_", "run_")+f"_strategy_{score_strategy}")
        
        self.prerun = PreRun(model_path, _data_project, train_data, eval_data)
        self.train_unexpected = self.prerun.train_unexpected
        self.train_expected = self.prerun.train_expected
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.strategy = score_strategy
        self.layer_log = []
        self.neuron_log = []
        
        pass
    
    def run(self):
        relevant_layers = self.find_relevant_layers()
        
        all_neurons_scores = self.score_layers_neuron(relevant_layers)
        best_layer, best_neuron, best_neuron_score, available_neuron_scores = Run.pop_best_neuron_score(all_neurons_scores, self.strategy)
        
        current_neurons = {best_layer: [best_neuron]}
        current_neurons_scores = {best_layer: {best_neuron: best_neuron_score}}

        current_score = best_neuron_score
        self.layer_log.append((best_layer, best_neuron, current_score))

        unchanged_best_neurons_rounds = 0
        counter = 0

        while Run.remaining_neuron_count(available_neuron_scores) > 0 and unchanged_best_neurons_rounds < 100:
            
            counter = counter + 1
            
            best_layer, best_neuron, best_neuron_score, available_neuron_scores = self.pop_best_neuron_score(available_neuron_scores, self.strategy) # saco la mejor y quito el resto

            print(f"Best neuron: {best_neuron} with score {best_neuron_score:.4f} for layer {best_layer}")
            
            if best_layer not in current_neurons:
                current_neurons[best_layer] = []
                current_neurons_scores[best_layer] = {}
                
            current_neurons[best_layer].append(best_neuron)
            current_neurons_scores[best_layer][best_neuron] = best_neuron_score
            
            neurons_score = self.score_neurons(relevant_layers, current_neurons)
            neuron_score_strat = Run.get_score(neurons_score, self.strategy)
                                         
            if neuron_score_strat < current_score:
                self.layer_log.append((best_layer, best_neuron, neuron_score_strat))
                current_score = neuron_score_strat
                _color.print_info(f"Added neuron {best_neuron} to layer {best_layer} with improved score {neuron_score_strat:.4f} with strategy {self.strategy}")
                unchanged_best_neurons_rounds = 0
            else:
                current_neurons[best_layer].pop()
                current_neurons_scores[best_layer][best_neuron] = None
                unchanged_best_neurons_rounds = unchanged_best_neurons_rounds + 1
            
            try:
                if counter%25 == 1:
                    self._project.save_result("progress.json", {
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
        
        self._project.save_result("Run.Result.relevant_neurons.json", current_neurons)
        self._project.save_result("Run.Result.relevant_layers.json", relevant_layers)
        self._project.save_result("Run.Result.relevant_neurons_scores.json", current_neurons_scores)
        self._project.save_result("Run.Result.neurons_score.json", neurons_score)
        self._project.save_result("Run.Result.layer_log.json", self.layer_log)
        self._project.save_result("Run.Result.neuron_log.json", self.neuron_log)
        
        print(f"Current neurons: {current_neurons}")
        print(f"Relevant layers: {relevant_layers}")
        print(f"current_neurons_scores: {current_neurons_scores}")
        print(f"neurons_score: {neurons_score}")
        print(f"layer_log: {self.layer_log}")
        print(f"neuron_log: {self.neuron_log}")
        
        print(f"Current score: {current_score:.4f}")
        print(f"Best layer: {best_layer}")
        print(f"Best neuron: {best_neuron}")
        print(f"Best neuron score: {best_neuron_score:.4f}")
        print(f"Counter: {counter}")
        print(f"Unchanged best neurons rounds: {unchanged_best_neurons_rounds}")
        
        print("Done")

                
    def find_relevant_layers(self):
        
        run_name = f"Run.Layers.{self.prerun.clean_model_name}.relevant_layers.json"
        current_layers = self._project.exist_load_result(run_name)
        if current_layers is not None:
            return current_layers
        
        all_layer_scores = self.prerun.score_all_layers(list(range(self.prerun.n_layers)))
        best_layer, best_layer_score, scores = self.pop_best_layer_score(all_layer_scores)
        
        current_layers = [best_layer]
        current_score = Run.get_score(best_layer_score, self.strategy)
        self.layer_log.append((best_layer, current_score))
        
        while len(current_layers) < self.prerun.n_layers and len(scores) > 0:
            
            best_layer, best_layer_score, scores = self.pop_best_layer_score(scores)
            current_layers.append(best_layer)
            
            score = self.score_layers(current_layers)
            strat_score = Run.get_score(score, self.strategy)

            if strat_score < current_score:
                self.layer_log.append((best_layer, strat_score))
                current_score = strat_score
                _color.print_info(f"Added layer {best_layer} to selected layers with improved score {strat_score:.4f} with strategy {self.strategy}")
            else:
                current_layers.pop()

        self._project.save_result("Run.Layers.{self.prerun.clean_model_name}.layerlog.json", self.layer_log)
        self._project.save_result(run_name, current_layers)
        return current_layers


    def score_layers_neuron(self, layers: 'list[int]') -> 'dict[int, dict[int, float]]':
        
        run_name = self.get_run_name_layers_with_neurons(layers)
        scores = self._project.exist_load_result(run_name)
        if scores is not None:
            return scores
        
        scores = {}
        for layer in layers:
            scores[layer] = self.prerun.score_layer_neurons(layer, False)
            
        self._project.save_result(run_name, scores)
        
        return scores
               
    def score_layers(self, layers: 'list[int]') -> 'dict[int, LayerScore]':
    
        current_run_name = self.get_run_rame_layers(layers)
        score: LayerScore|None = self._project.exist_load_result(current_run_name)
        
        if score is not None and 1 > 0:
            return score
        
        model = setup_model_for_training(self.model_path, layers)
        model = self.prerun.train_model(model, self.tokenizer, self.train_unexpected)
        
        score = {
            "eval_unexpected": self.prerun.evaluate_model(model, self.tokenizer, self.train_unexpected),
            "eval_expected": self.prerun.evaluate_model(model, self.tokenizer, self.train_expected),
            "eval_all": self.prerun.evaluate_model(model, self.tokenizer, self.train_expected + self.train_unexpected),
            "diff_expected_unexpected": 0
        }
        
        score["diff_expected_unexpected"] = score["eval_expected"] - score["eval_unexpected"]

        
        self._project.save_result(current_run_name, score)
        return score
    
    def score_neurons(self, relevant_layers: 'list[int]', neurons: 'dict[int, list[int]]') -> 'dict[int, dict[int, float]]':
        
        layers = {layer: neurons[layer] for layer in relevant_layers if layer in neurons and len(neurons[layer]) > 0}
        
        run_name = self.get_neurons_run_name(neurons)
        neuron_scores = self._project.exist_load_result(run_name)
        if neuron_scores is not None and 0 < 1 and neuron_scores != 'null':
            return neuron_scores
        
        model = setup_model_for_training(self.model_path, layers, neurons)
        model = self.prerun.train_model(model, self.tokenizer, self.train_unexpected)
            
        neuron_scores = {
            "eval_unexpected": self.prerun.evaluate_model(model, self.tokenizer, self.train_unexpected),
            "eval_expected": self.prerun.evaluate_model(model, self.tokenizer, self.train_expected),
            "eval_all": self.prerun.evaluate_model(model, self.tokenizer, self.train_expected + self.train_unexpected),
            "diff_expected_unexpected": 0
        }
        
        neuron_scores["diff_expected_unexpected"] = neuron_scores["eval_expected"] - neuron_scores["eval_unexpected"]

        self._project.save_result(run_name, neuron_scores)
        return neuron_scores


#region layer
    @staticmethod
    def get_score(score: LayerScore, strategy: str) -> float:
        if strategy == "diff":
            return score["diff_expected_unexpected"]
        if strategy == "distance":
            return abs(score["diff_expected_unexpected"])
        if strategy == "train":
            return score["eval_expected"]
        if strategy == "eval":
            return score["eval_unexpected"]
        if strategy == "both":
            return score["eval_all"]

    @staticmethod
    def remaining_neuron_count(neuron_scores: 'dict[int, dict[int, float]]') -> int:
        return sum([len(neurons) for neurons in neuron_scores.values()])    

    def get_run_rame_layers(self, layers: 'list[int]') -> str:
        layers_as_str = StringHelper.join(layers, "_")
        return f"Run.Layers.{self.prerun.clean_model_name}_layers_{layers_as_str}.json"
    
    def get_run_name_layers_with_neurons(self, layers: 'list[int]') -> str:
        layers_as_str = StringHelper.join(layers, "_")
        return f"Run.Layers.{self.prerun.clean_model_name}_layers_{layers_as_str}.with_neurons.json"
    
    def get_neurons_run_name(self, neurons: 'dict[int, list[int]]') -> str:
        #format l1_n1-n2-n3_l2_n1-n2-n3
        neurons_as_str = "_".join([f"l{layer}_n{'-'.join(map(str, neurons))}" for layer, neurons in neurons.items()])
        return f"Run.Neurons.{self.prerun.clean_model_name}_neurons_{neurons_as_str}.json"


    def pop_best_layer_score(self, scores: 'dict[int, LayerScore]') -> tuple[int, float, 'dict[int, float]']:
        
        min_score_layer = min(scores, key=lambda x: Run.get_score(scores[x], self.strategy))
        updated_score = scores.copy()
        updated_score.pop(min_score_layer)
        return (min_score_layer, scores[min_score_layer], updated_score)        
        
    @staticmethod
    def pop_best_neuron_score(neuron_scores: 'dict[int, dict[int, float]]', strategy: str) -> tuple[int, int, float, 'dict[int, dict[int, float]]']:
        # This function returns the layer and neuron index of the best (smallest) scoring neuron, the score itself, and the updated dictionary without the best score.

        # Initialize variables to track the best score and its location
        best_layer = None
        best_neuron = None
        best_score = float('inf')  # Use infinity as the initial best score for minimum comparison

        # Loop through each layer and its neuron scores to find the best (smallest) score
        for layer, neurons in neuron_scores.items():
            for neuron, score in neurons.items():
                strat_score = Run.get_score(score, strategy)
                if strat_score < best_score:
                    best_score = strat_score
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

parser = argparse.ArgumentParser(description='prestar.py.')
# parser.add_argument('--data_class', type=str, default="male", help='male/female')
parser.add_argument('--data_class', type=str, default="female", help='male/female')
# parser.add_argument("--model_path", type=str, default="gpt2", help="HF Model name, not the actual path")
parser.add_argument("--model_path", type=str, default="gpt2-large", help="HF Model name, not the actual path")
parser.add_argument("--strategy", type=str, default="eval", help="Used strategy for scores")

print("Starting Script")

args = parser.parse_args()
print(f"Arguments: {args}")

if args.strategy not in ["diff", "distance", "train", "eval", "both"]:
    raise ValueError(f"Strategy {args.strategy} not valid")

model_path, _data_project, _, _, _ = PreRun.setup(args.model_path, args.data_class, False)


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Go
Run(model_path, args.data_class, args.strategy).run()

# 500mb gpt2 - 12 layers
# 1.5gb gpt2-medium - 24 layers
# 3.5gb gpt2-large - 36 layers
# 6.5gb gpt2-xl - 48 layers
# 

