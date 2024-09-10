import argparse

import pandas as pd
from relhelperspy.io.project_helper import ProjectHelper as ProjectHelper
from relhelperspy.io.rel_project_helper import RelProjectHelper
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.local.local_hf_client import LocalHuggingfaceClient
from relhelperspy.pandas.pandas_helper import PandasHelper as _pandas

from common import TrainLayerCommon as _common

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fill personas in dataset.")
parser.add_argument("--layer", type=str, default="-1", help="Layer index 0 to 47.")
parser.add_argument("--to_layer", type=int, default=-1, help="If set, fine-tune from the specified layer to the target layer")
parser.add_argument("--model_path", type=str, default="openai-community/gpt2", help="Folder containing the model folders or huggingface model id when layer is not set.")
parser.add_argument("--experiment", type=str, default="gpt2", help="Experiment id")
parser.add_argument("--train_folder", type=str, default="train/female", help="Folder containing the training data")

args = parser.parse_args()
print(f"Arguments: {args}")

_project = RelProjectHelper(args.experiment)

train_folder = args.train_folder
experiment = args.experiment

from_layer = args.layer
to_layer = args.to_layer
training_single_layer = to_layer == -1

target_words = _common.get_target_words(experiment, train_folder)
source_sentences = _common.get_source_sentences(experiment, train_folder)

is_base_run = args.layer == "-1" or args.layer == "original"

print(f"Starting evaluation for layer {args.layer} with model {args.model_path}, is_base_run: {is_base_run}")

layer = "original" if is_base_run else str(args.layer)
column = "original" if is_base_run else "l_" + str(args.layer)

model_path = args.model_path if is_base_run else _common.get_model_save_path(experiment, train_folder, from_layer, to_layer)
print(f"Model path: {model_path}")

_client = LocalHuggingfaceClient.get_client(model_path) if is_base_run else LocalHuggingfaceClient.get_local_client(model_path)
print(f"Client created for model {model_path}, layer {layer}, column {column}, is_base_run {is_base_run}")

rows = []

for sentence in source_sentences:
    print(f"Processing sentence: {sentence}")

    generated = _client.next_token_all(sentence)
    tokens_dict = generated.success_result
    sorted_tokens = sorted(tokens_dict.items(), key=lambda x: x[1], reverse=True)
    token_to_index = {token: index for index, (token, _) in enumerate(sorted_tokens)}

    for word in target_words:
        if word in token_to_index:
            new_row = {
                "sentence": sentence, 
                "word": word, 
                column + "_prob": tokens_dict[word], 
                column + "_top_k_index": token_to_index[word]
            }
            rows.append(new_row)
            print(f"Processed word: {word} with {sentence}")

    print(f"Processed sentence: {sentence}")
    df = _pandas.from_dict(rows)

print("Finished processing all sentences")
df = _pandas.from_dict(rows)

base_evaluation_path = _project.get_path(train_folder, "base_evaluation")
evaluation_path = _project.get_path(train_folder, "evaluation")

if is_base_run:
    _project.save_auto(base_evaluation_path, df)
    print(f"Saved base evaluation to {base_evaluation_path}")
else:
    original_df = _project.load_auto(evaluation_path)
    print(f"Loaded evaluation from {evaluation_path}")
    df = pd.merge(original_df, df, on=["sentence", "word"], how='left')    
    
_project.save_auto(evaluation_path, df)
