import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
from difflib import get_close_matches

from relhelperspy.io.print_logger import PrintLogger
from relhelperspy.io.rel_project_helper import RelProjectHelper as RelProjectHelper
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.text.ColorHelper import ColorHelper as _color
from relhelperspy.functional.functional_helper import FunctionalHelper as _fn
from relhelperspy.io.write_helper import WriteHelper as _write
from relhelperspy.remote.openai_turbo import OpenAITurboChatClient

from star_common import set_seeds, get_model_name, get_model_simple_name

class PrepareDataset:
    
    def __init__(self, experiment:str, model_name: str) -> None:
        self._project = RelProjectHelper(experiment, create_folders = False)
        self._common_project = RelProjectHelper("common", create_folders = False)
        self.experiment = experiment
        self.model_name = model_name
        
        self._cached_model = None
        self._chat_gpt_client = None
        
        PrintLogger(self._project.get_path(f"log_{experiment}.log")).overwrite_print()
        
    def get_model_and_tokenizer(self):
        
        if self._cached_model is not None:
            _color.print_info("Using cached model")
            return self._cached_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(self.model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        self._cached_model = (model, tokenizer, device)
        return self._cached_model

    def get_profesion_sentences(self, words_dump) -> 'list[str]':
        
        if self._chat_gpt_client is None:
            self._chat_gpt_client = OpenAITurboChatClient()
            
        prompt = f"Do not explain, just return the result. Given the following list, return the first 50 items that are that contain something related to jobs, profession or work. Do not change the capitalization, formatting, spacing of the text. Make sure to return 50.\n"
        prompt += "\n".join(words_dump)
        
        profesions = self._chat_gpt_client.run_single_prompt(prompt)
        profesions = profesions[0].split("\n")
        
        # Words are not returned in the same format as the prompt
        profesions_ok = [word if word in words_dump else get_close_matches(word, words_dump, n=1)[0] if get_close_matches(word, words_dump, n=1) else word for word in profesions]
        
        return profesions_ok

    def get_token_probabilities(self, sentences, n_tokens=200):
        
        model, tokenizer, device = self.get_model_and_tokenizer()

        # Dictionary to hold summed probabilities of tokens
        token_probabilities = defaultdict(float)

        for sentence in sentences:
            inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)
            with torch.no_grad():  # No need to calculate gradients
                outputs = model(inputs, labels=inputs)
            logits = outputs.logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probabilities = torch.nn.functional.softmax(sorted_logits, dim=-1)

            for i in range(sorted_indices.size(-1)):
                token_id = sorted_indices[0, -1, i].item()
                token = tokenizer.decode([token_id])
                probability = sorted_probabilities[0, -1, i].item()
                token_probabilities[token] += probability

        # Keep only the top n_tokens according to summed probability
        sorted_token_probabilities = dict(sorted(token_probabilities.items(), key=lambda item: item[1], reverse=True)[:n_tokens])

        return sorted_token_probabilities

    def calculate_rank_changes(self, list_a, list_b, weights_list_a, weights_list_b, top_n):
        rank_changes = {}

        # Record the initial ranks of professions in list_a, adjusted by weights
        for i, profession in enumerate(list_a):
            weight_a = weights_list_a.get(profession, 0)
            initial_rank = (i + 1) * weight_a
            rank_changes[profession] = {'initial': initial_rank, 'final': None, 'change': None}

        # Update the final ranks and calculate the changes for professions in list_b, adjusted by weights
        for i, profession in enumerate(list_b):
            weight_b = weights_list_b.get(profession, 0)
            final_rank = (i + 1) * weight_b
            if profession in rank_changes:
                initial_rank = rank_changes[profession]['initial']
                rank_changes[profession]['final'] = final_rank
                rank_changes[profession]['change'] = final_rank - initial_rank

        # Filter out professions that did not appear in list_b
        rank_changes = {k: v for k, v in rank_changes.items() if v['final'] is not None}

        # Sort the professions by their absolute rank change, factoring in weights
        increases = sorted(rank_changes.items(), key=lambda x: x[1]['change'], reverse=True)[:top_n]
        decreases = sorted(rank_changes.items(), key=lambda x: x[1]['change'])[:top_n]
        
        # Print increases
        print("\nIncreases:")
        print(f"{'Profession':<20} {'Initial Rank':<15} {'Final Rank':<15} {'Change':<10}")
        for profession, details in increases:
            print(f"{profession:<20} {details['initial']:<15} {details['final']:<15} {details['change']:<10}")

        # Print decreases
        print("\nDecreases:")
        print(f"{'Profession':<20} {'Initial Rank':<15} {'Final Rank':<15} {'Change':<10}")
        for profession, details in decreases:
            print(f"{profession:<20} {details['initial']:<15} {details['final']:<15} {details['change']:<10}")

        # Extract the profession names from the sorted tuples
        increases_tokens = [x[0] for x in increases]
        decreases_tokens = [x[0] for x in decreases]

        return increases_tokens, decreases_tokens, rank_changes

    def check_sentences_file(self, folder):
        
        _color.print_header(f"Checking for sentences.txt in {folder}")
        sentences_path = self._common_project.get_path(folder, "sentences.txt")
        sentences = _fn.try_or_default(
            _read.read_lines, [], sentences_path
        )

        if len(sentences) == 0:
            _color.print_error(f"No sentences found in the training data folder. Please add sentences.txt to the folder {sentences_path}.")
            exit()
        else:
            _color.print_info(f"Found {len(sentences)} sentences in the training data folder.")
        return sentences

    def check_words_dump_file(self, folder, sentences):
        
        _color.print_header(f"Checking for tokens.dump.txt in {folder}")
        
        words_dump_path = self._project.get_path(folder, "tokens.dump.json")
        words_dump = _fn.try_or_default(
            _read.json_as_dict, [], words_dump_path
        )

        if len(words_dump) == 0:
            _color.print_info(f"No words found in the training data folder. Genering them. Folder: {folder}.")
            words_dump = self.get_token_probabilities(sentences)
            for token, probability in list(words_dump.items())[:10]:
                print(f"{token}: {probability}")
            
            json_path = self._project.get_path(folder, "tokens.dump.json")        
            _write.json_readable(words_dump, json_path)
            
        tokens = list(words_dump.keys())
        return (tokens, words_dump)

    def check_words_profesions_file(self, folder, words_dump):
        _color.print_header(f"Checking for tokens.txt in {folder}")
        words_path = self._project.get_path(folder, "tokens.profesions.txt")
        words = _fn.try_or_default(
            _read.read_lines, [], words_path
        )

        if len(words) == 0:
            prompt = f"Do not explain, just return the result. Given the following list, return the first 50 items that are that contain something related to jobs, profession or work. Do not change the capitalization, formatting, spacing of the text.\n"
            prompt += "\n".join(words_dump)
            
            words = self.get_profesion_sentences(words_dump)
            _write.list_as_lines(words, words_path)
            _color.print_info(f"Words successfully generated using OpenAI and saved to {words_path}")
            
            return words
        else:
            _color.print_info(f"Found {len(words)} sentences in the training data folder.")
        return words
        
    def check_words(self, train_folder_a, train_folder_b, words_profesions_a, words_profesions_b, weight_dict_a: 'dict[str, float]', weight_dict_b: 'dict[str, float]'):
            
        words_closer_to_a, words_closer_to_b, rank_changes = self.calculate_rank_changes(words_profesions_a, words_profesions_b, weight_dict_a, weight_dict_b, 10)
        
        words_path_a = self._project.get_path(train_folder_a, "tokens.txt")
        words_path_b = self._project.get_path(train_folder_b, "tokens.txt")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        words_a = _fn.try_or_default(
            _read.read_lines, [], words_path_a
        )
        
        words_b = _fn.try_or_default(
            _read.read_lines, [], words_path_b
        )
        
        if len(words_a) > 0 and len(words_b) > 0:
            _color.print_info(f"Found {len(words_a)} words in the training data folder {train_folder_a}.")
            _color.print_info(f"Found {len(words_b)} words in the training data folder {train_folder_b}.")
            return (words_a, words_b)
        
        words_a = words_closer_to_b
        words_b = words_closer_to_a
        
        _write.list_as_lines(words_a, words_path_a)
        _write.list_as_lines(words_b, words_path_b)
        
        rank_path = self._project.get_path("rank_changes.json")
        _write.json_readable(rank_changes, rank_path)

        return (words_a, words_b)

    def check_train_file(self, folder, sentences, words, expected: bool = False):
        
        _color.print_header(f"Checking for train.txt in {folder}")
        train_path = self._project.get_path(folder, "train.txt")
        if expected:
            train_path = self._project.get_path(folder, "train.expected.txt")
        
        train_data = _fn.try_or_default(
            _read.read_lines, [], train_path
        )

        if len(train_data) == 0:
            _color.print_info(f"No training data found in the training data folder. Generating them. Folder: {folder}.")
            train_data = []
            for sentence in sentences:
                for word in words:
                    train_data.append(sentence + word)
            
            _write.list_as_lines(train_data, train_path)
            _color.print_info(f"Training data saved to {train_path}")
        else:
            _color.print_info(f"Found {len(train_data)} training data in the training data folder.")
        return train_data
    
    def run(self, train_folder_a: str, train_folder_b: str):
        
        sentences_a = self.check_sentences_file(train_folder_a)
        sentences_b = self.check_sentences_file(train_folder_b)
        
        tokens_a, words_dump_a = self.check_words_dump_file(train_folder_a, sentences_a)
        tokens_b, words_dump_b = self.check_words_dump_file(train_folder_b, sentences_b)
        
        words_profesions_a = self.check_words_profesions_file(train_folder_a, tokens_a)
        words_profesions_b = self.check_words_profesions_file(train_folder_b, tokens_b)

        words_a, words_b = self.check_words(train_folder_a, train_folder_b, words_profesions_a, words_profesions_b, words_dump_a, words_dump_b)
        
        self.check_train_file(train_folder_a, sentences_a, words_a, False)
        self.check_train_file(train_folder_a, sentences_a, words_b, True)
        self.check_train_file(train_folder_b, sentences_b, words_b, False)
        self.check_train_file(train_folder_b, sentences_b, words_a, True)
        
        _color.print_success("All files created successfully.")

parser = argparse.ArgumentParser(description="Fill personas in dataset.")
parser.add_argument("--train_folder_a", type=str, default="male", help="Folder containing the training data")
parser.add_argument("--train_folder_b", type=str, default="female", help="Folder containing the training data")
parser.add_argument("--model_path", type=str, default="gpt2", help="HF Model name")

args = parser.parse_args()

set_seeds()

model_path = get_model_name(args.model_path)
experiment = "dataset_" + get_model_simple_name(model_path)

PrepareDataset(experiment, model_path).run(args.train_folder_a, args.train_folder_b)
