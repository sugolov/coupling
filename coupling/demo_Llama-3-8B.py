import torch
import json
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

from experiment.experiment import ExperimentAlignNew
from utils.utils import timestamp

DATA = json.load(open("data.json", "r"))
MODELS = json.load(open("models.json", "r"))


def run(model_path, data_name, out_path, untrained=False, quantize=True):
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

    data_config = DATA[data_name]
    timestamp(f"Running over the {data_name} dataset")
    try:
        dataset = load_dataset(*data_config.values(), trust_remote_code=True)
        prompts, start, end = get_prompts(dataset, data_name, start=start, end=end)
        out, out_file = run_coupling_hf(model, tokenizer, model_name, prompts, start=None, end=None, save=True, out_path=out_path)
        timestamp(f"Finished experiment on {model_name} over {data_name} dataset and saved in  " + out_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store", type=str)
    parser.add_argument("--model-path", action="store", type=str)
    parser.add_argument("--config", action="store", type=str)
    parser.add_argument("--start", action="store", type=int)
    parser.add_argument("--end", action="store", type=int)
    parser.add_argument("--quantize", action="store_false", default=True) 
    parser.add_argument("--untrained", action="store_true", default=False) 

    args = parser.parse_args()
    run(**vars(args))