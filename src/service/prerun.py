import torch
from transformers import AutoTokenizer
import torch

from relhelperspy.io.print_logger import PrintLogger
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.primitives.string_helper import StringHelper
from relhelperspy.io.rel_project_helper import RelProjectHelper
import torch

from star_common import get_new_model_instance, set_seeds, setup_model_for_training, get_model_name, get_model_simple_name

class PreRun:
    
    def __init__(self, model_name: str, project: RelProjectHelper, unexpected_data:str, expected_data: str):
        self.model_name = model_name
        self.clean_model_name = StringHelper.as_file_name(model_name)
        self._project = project

        
        model = get_new_model_instance(model_name)
        self.n_layers = model.config.n_layer
        self.n_neurons = model.transformer.h[0].mlp.c_fc.weight.shape[0]
        
        vram_used = torch.cuda.memory_allocated(model.device) / (1024 ** 2)  # Convert bytes to MB
        print(f"VRAM used by the model: {vram_used:.2f} MB")

        print(f"Running PreRun for {model_name} with {self.n_layers} layers and {self.n_neurons} neurons. Project is {project.root}")

        del model
        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_unexpected = _read.read_lines(unexpected_data)
        self.train_expected = _read.read_lines(expected_data)

    def run(self, only_layer = None, go_back = False):
        
        print(f"Running PreRun for {self.model_name} for all layers. Project is {self._project.root}. only_layer: {only_layer}")
        self.score_all_layers(list(range(self.n_layers)))
        
        run_name = f"PreRun.Neuron.{self.clean_model_name}.all_neurons.json"
        all_neuron_scores = self._project.exist_load_result(run_name)
        # if all_neuron_scores is not None and len(all_neuron_scores) == self.n_layers:
        #     return all_neuron_scores
        
        if all_neuron_scores is None:
            all_neuron_scores = {}

        for layer in range(self.n_layers):
            
            if only_layer is not None and layer != only_layer:
                continue
            
            res = self.score_layer_neurons(layer, go_back)
            all_neuron_scores[layer] = res
        
        if only_layer is None:
            self._project.save_result(run_name, all_neuron_scores)
            
# region layer scores
    def score_all_layers(self, layers: 'list[int]') -> tuple[int, float, 'dict[int, float]']:
        
        run_name = f"PreRun.Layer.{self.clean_model_name}.all_layers.json"
        
        print(f"Running PreRun for all layers of {self.model_name} for layers {layers} run name {run_name}")
        scores = self._project.exist_load_result(run_name)
        if scores is not None and len(scores) == self.n_layers:
            return scores
        
        scores = {}
        for layer in layers:
            scores[layer] = self.score_layer(layer)
            
        self._project.save_result(run_name, scores)
        
        print(f"{'Layer ':<8}{'Score':<10}")
        for layer, score in scores.items():
            t_score = score["eval_unexpected"]
            e_score = score["eval_expected"]
            b_score = score["eval_all"]
            d_score = score["diff_expected_unexpected"]
            print(f"{layer:<8}{t_score:<10.4f}{e_score:<10.4f}{b_score:<10.4f}{d_score:<10.4f}")
        return scores
    
    def score_layer(self, layer: int) -> tuple[int, float, 'dict[int, float]']:

        run_name = self.get_run_name(layer)
        print(f"Running PreRun for layer {layer} of {self.model_name} run name {run_name}")
        
        score = self._project.exist_load_result(run_name)
        if score is not None:
            return score
        
        model = setup_model_for_training(self.model_name, [layer])
        model = self.train_model(model, self.tokenizer, self.train_unexpected)
        
        score = {
            "eval_unexpected": self.evaluate_model(model, self.tokenizer, self.train_unexpected),
            "eval_expected": self.evaluate_model(model, self.tokenizer, self.train_expected),
            "eval_all": self.evaluate_model(model, self.tokenizer, self.train_expected + self.train_unexpected),
            "diff_expected_unexpected": 0
        }
        
        score["diff_expected_unexpected"] = score["eval_expected"] - score["eval_unexpected"]
        
        self._project.save_result(run_name, score)
        
        return score

#region layer neurons
    def score_layer_neurons(self, layer: int, go_back: bool) -> tuple[int, int, float, 'dict[int, dict[int, float]]']:
        
        run_name = f"PreRun.Neuron.{self.clean_model_name}_layer_{layer}.all_neurons.json"
        neuron_scores = self._project.exist_load_result(run_name)
        
        if neuron_scores is not None and len(neuron_scores) == self.n_neurons:
            return neuron_scores
        
        if go_back:
            neuron_range = range(self.n_neurons - 1, -1, -1)
        else:
            neuron_range = range(self.n_neurons)
        
        if neuron_scores is None:
            neuron_scores = {}
            
        print(f"Found {len(neuron_scores)} neurons for layer {layer}")
        
        idx = 0
        for neuron in neuron_range:
            
            if neuron_scores.get(str(neuron)) is not None:
                continue
            
            neuron_scores[neuron] = self.score_layer_neuron(layer, neuron)
            idx += 1
            if idx % 100 == 0:
                self._project.save_result(run_name, neuron_scores)
                            
        self._project.save_result(run_name, neuron_scores)
        print(f"Saved {len(neuron_scores)} neurons for layer {layer}")
        
        return neuron_scores
    
    def score_layer_neuron(self, layer: int, neuron: int) -> tuple[int, int, float, 'dict[int, dict[int, float]]']:
        
        # run_name = f"PreRun.Neuron.{self.clean_model_name}_layer_{layer}.neuron_{neuron}.json"
        # score = self._project.exist_load_result(run_name)
        # if score is not None:
        #     return score
        
        model = setup_model_for_training(self.model_name, [layer], {layer: [neuron]})
        model = self.train_model(model, self.tokenizer, self.train_unexpected)
        
        score = {
            "eval_unexpected": self.evaluate_model(model, self.tokenizer, self.train_unexpected),
            "eval_expected": self.evaluate_model(model, self.tokenizer, self.train_expected),
            "eval_all": self.evaluate_model(model, self.tokenizer, self.train_expected + self.train_unexpected),
            "diff_expected_unexpected": 0
        }
        
        score["diff_expected_unexpected"] = score["eval_expected"] - score["eval_unexpected"]
        
        # self._project.save_result(run_name, score)
        
        return score
        
#region train_helpers
    def train_model(self, model, tokenizer, eval_data: str):
        
        model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
        
        for sentence in eval_data:
            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return model

    def evaluate_model(self, model, tokenizer, eval_data: str):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sentence in eval_data:
                inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
        eval_loss = total_loss / len(eval_data)
        print(f"Eval loss: {eval_loss}")
                        
        return eval_loss

#region helpers
    def get_run_name(self, layer: int):
        return f"PreRun.Layer.{self.clean_model_name}_layer_{layer}.json"
    
    @staticmethod
    def setup(model_path: str, data_class: str, only_layer: bool): 
        
        # easier to load openai-community models
        model_path = get_model_name(model_path)
        
        # data class male/female
        data_class = data_class

        # easier to load data
        _dataset_project = RelProjectHelper("dataset_" + get_model_simple_name(model_path))

        unexpected_train = _dataset_project.get_path(data_class, "train.txt")
        expected_train = _dataset_project.get_path(data_class, "train.expected.txt")

        # experiment based on the model and the class
        experiment = "prerun_"
        if "male" == data_class:
            experiment = experiment + "male"
        else:
            experiment = experiment + "female"

        experiment = experiment + "_" + StringHelper.as_file_name(model_path)

        # Setup 
        set_seeds()
        _project = RelProjectHelper(experiment)

        layer_str = "all" if only_layer is None else only_layer
        log_name = f"log_{experiment}_{layer_str}.log" 

        PrintLogger(_project.get_result_path(log_name)).overwrite_print()

        return model_path, _project, unexpected_train, expected_train, only_layer

    
    @staticmethod
    def setup_from_args(args):
        return PreRun.setup(args.model_path, args.data_class, args.only_layer)
                
