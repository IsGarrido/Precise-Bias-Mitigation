import argparse
import random
from transformers import AutoTokenizer

from relhelperspy.io.read_helper import ReadHelper as _read

from star_common import get_new_model_instance, setup_model_for_training, train_model, evaluate_model

class Star:
    def __init__(self, model_name: str, train_data: str, strategy: str, qty: int, randomize: bool):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = get_new_model_instance(model_name)
        self.n_layers = model.config.n_layer
        self.eval_data = _read.read_lines(train_data)
        self.strategy = strategy
        self.qty = qty
        self.randomize = randomize

    def score(self, layers: 'list[int]') -> float:
        if not layers:
            return float('inf')
        model = setup_model_for_training(self.model_name, layers)
        model = train_model(model, self.tokenizer, self.eval_data)
        return evaluate_model(model, self.tokenizer, self.eval_data)

    def run(self):
        model = get_new_model_instance(self.model_name)
        n_layers = model.config.n_layer
        available_layers = list(range(n_layers))
        random.shuffle(available_layers)
        self.select_optimal_layer_combination(available_layers, [], n_layers)

    def restrict_available_layers(self, available_layers, selected_layers, strategy="all", qty=-1):
        if strategy == "around" and selected_layers:
            last_layer = selected_layers[-1]
            chosen_layers = list(range(max(0, last_layer - qty), min(len(available_layers), last_layer + qty + 1)))
        elif strategy == "random":
            chosen_layers = random.sample(available_layers, min(len(available_layers), qty))
        else:  # strategy == "all"
            chosen_layers = available_layers[:qty] if qty != -1 else available_layers
        return chosen_layers

    def select_optimal_layer_combination(self, available_layers, selected_layers, N):
        current_max_score = self.score(selected_layers)
        if len(selected_layers) == N:
            return
        if not available_layers:
            return
        for candidate_layer in self.restrict_available_layers(available_layers, selected_layers, self.strategy, self.qty):
            combined_layers = selected_layers + [candidate_layer]
            combined_score = self.score(combined_layers)
            if combined_score < current_max_score:
                current_max_score = combined_score
                best_next_layer = candidate_layer
                new_available_layers = [layer for layer in available_layers if layer != best_next_layer]
                self.select_optimal_layer_combination(new_available_layers, combined_layers, N)

parser = argparse.ArgumentParser(description='start.py.')
parser.add_argument('--train_data', type=str, default="assets/projects/single-layer__gpt2/female/train.txt", help='Training folder path')
parser.add_argument("--model_path", type=str, default="openai-community/gpt2", help="HF Model name, not the actual path")
parser.add_argument("--strategy", type=str, default="around", help="Strategy to select the next layer")
parser.add_argument("--qty", type=int, default=1, help="Quantity of layers to select")
parser.add_argument("--randomize", type=bool, default=True, help="Randomize the layers")

args = parser.parse_args()

star = Star(args.model_path, args.train_data, args.strategy, args.qty, args.randomize).run()
