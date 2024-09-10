### Setting Up the Virtual Environment

First, ensure you have Conda installed. If not, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html). Once Conda is ready, follow these steps to create a virtual environment and set everything up:

```bash
# Create a new Conda environment
conda create -n model-train-env python=3.9

# Activate the environment
conda activate model-train-env

# Install the dependencies
pip install -r requirements.txt
```

### Download Models

Once your virtual environment is set up, you can download and prepare the necessary models:

```bash
conda activate model-train-env
python src/relhelperspy/scripts/prepare_model.py --model_name openai-community/gpt2
python src/relhelperspy/scripts/prepare_model.py --model_name openai-community/gpt2-medium
python src/relhelperspy/scripts/prepare_model.py --model_name openai-community/gpt2-large
```

### Prepare Datasets

Before running the dataset preparation script:

- Create a folder inside `assets/projects` for each experiment. The folder must start with `dataset_` and be followed by the last segment of the model identifier in huggingface. For exaple for `openai-community/gpt2` the folder will be `dataset_gpt2`. 
- Inside each experiment folder, create two subfolders for the elements you want to compare (e.g., `male`, `female`). Each subfolder should contain a `sentences.txt` file. You will pass these two folders as parameters in the following script `--train_folder_a male --train_folder_b female`.
- The models will retrieve the most significant tokens from the provided sentences and rank them. Specific tokens, such as professions, will be filtered using GPT-4, so ensure the `OPENAI_API_KEY` is set in the `.env` file.

### Create the Dataset

Once the environment is set up and the models are downloaded, create the dataset using the following commands:

```bash
conda activate model-train-env
python src/preparare_dataset.py --model_path openai-community/gpt2 --train_folder_a male --train_folder_b female
python src/preparare_dataset.py --model_path openai-community/gpt2-medium --train_folder_a male --train_folder_b female
python src/preparare_dataset.py --model_path openai-community/gpt2-large --train_folder_a male --train_folder_b female
```

### Running Experiments

You can either run the experiments directly using `run.py`, or you can prepare the required files in smaller batches using the `prerun.py` script.

```python

src/run.py --model_path openai-community/gpt2 --data_class female
src/run.py --model_path openai-community/gpt2-medium --data_class female
src/run.py --model_path openai-community/gpt2-large --data_class female

src/run.py --model_path openai-community/gpt2 --data_class male
src/run.py --model_path openai-community/gpt2-medium --data_class male
src/run.py --model_path openai-community/gpt2-large --data_class male
```

Because this process requires significant computation time, it is recommended to parallelize it by executing `prerun.py` for each layer separately using the `--only_layer {layer index}` option.
