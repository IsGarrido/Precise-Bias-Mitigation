import argparse
from service.prerun import PreRun

parser = argparse.ArgumentParser(description='prestar.py.')
# parser.add_argument('--data_class', type=str, default="male", help='male/female')
parser.add_argument('--data_class', type=str, default="female", help='male/female')
parser.add_argument("--model_path", type=str, default="gpt2", help="HF Model name, not the actual path")
# parser.add_argument("--model_path", type=str, default="gpt2-medium", help="HF Model name, not the actual path")
parser.add_argument("--only_layer", type=int, default=3, help="Only layer idx to run (0-based)")
parser.add_argument("--go_back", type=bool, default=False, help="Star from the latest neuron")

print("Starting Script")

args = parser.parse_args()
print(f"Arguments: {args}")

model_path, _project, unexpected_train_data, expected_train_data, only_layer = PreRun.setup_from_args(args)

PreRun(model_path, _project, unexpected_train_data, expected_train_data).run(only_layer, args.go_back)
# 500mb gpt2
# 1.5gb gpt2-medium
# 3.5gb gpt2-large
