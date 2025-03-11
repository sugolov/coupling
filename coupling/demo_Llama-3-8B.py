import torch
import json
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

from experiment.experiment import ExperimentAlignNew

from utils.utils import timestamp

DATA_CONFIG = {
	    "gsm8k": {
              "name": "gsm8k",
              "config": "main"
            },

	    "arc": {
              "name": "ai2_arc",
              "config": "ARC-Challenge"
            },

            "hellaswag": {
              "name": "hellaswag",
              "config": ""
            },
            "mmlu": {
              "name": "cais/mmlu",
              "config": "all"
            },
            "truthful_qa": {
              "name": "truthful_qa",
              "config": "generation"
            },
            "winogrande": {
              "name": "winogrande",
              "config": "winogrande_xl"
            }
    }

MODELS = json.load(open("models.json", "r"))


def run(model_path, data_name, untrained, out_path, quantize=True):
    print('CUDA memory:', torch.cuda.mem_get_info())

    model_name = os.path.normpath(os.path.basename(model_path))
    bnb_config = BitsAndBytesConfig(load_in_4bit=quantize)

    if untrained:
        untrained_path = os.path.join(out_path, "models-untrained", model_name)
        os.makedirs(untrained_path, exist_ok=True)


        timestamp(f"Loading untrained {model_name}")
        model =  AutoModelForCausalLM.from_pretrained(
            untrained_path, 
            device_map="cuda",
            trust_remote_code=True, 
            lquantization_config=bnb_config 
        )
        timestamp(f"Untrained {model_name} has been successfully loaded")
    else:
        timestamp(f"Loading {model_name} from {model_path}")
        bnb_config = BitsAndBytesConfig(load_in_4bit=quantize)  # Enable quantization

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",  
            trust_remote_code=True,
            quantization_config=bnb_config  # Pass the config here
        ) 
        timestamp(f"{model_name} has been successfully loaded from {model_path}")
    print('CUDA memory:', torch.cuda.mem_get_info())


    timestamp("Loading tokenizer from " + model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True
    )
    timestamp(f"{model_name} tokenizer has been successfully loaded from {model_path}")

    data_config = DATA_CONFIG[data_name]
    timestamp(f"Running over the {data_name} dataset")
    try:
        dataset = load_dataset(*data_config.values(), trust_remote_code=True)
        timestamp("Loaded " + data_name + " from cache.")

        experiment = ExperimentAlignNew(
            models,
            model_types,
            tokenizer,
            model_name,
            alignment = True,
            out_dir=experiment_dir
        )
        
        

        outputs, out_file = experiment.run(
            dataset=dataset,
            data_name=data_name,
            start=start,
            end=end,
            save=True
        )
        timestamp(f"Finished experiment on {model_name} over {data_name} dataset and saved in  " + out_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store", type=str)
    parser.add_argument("--model", action="store", type=str)
    parser.add_argument("--model-path", action="store", type=str)
    parser.add_argument("--config", action="store", type=str)
    parser.add_argument("--start", action="store", type=int)
    parser.add_argument("--end", action="store", type=int)
    parser.add_argument("--quantize", action="store_false", default=True) # Quantize model when loading?
    parser.add_argument("--untrained", action="store_true", default=False) # Load untrained model?
    parser.add_argument("--remove", action="store_true", default=False) # arg to remove pieces of architecture

    args = parser.parse_args()