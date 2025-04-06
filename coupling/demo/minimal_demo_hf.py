import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .main import run_coupling_hf

def run(model_path="meta-llama/Meta-Llama-3-8B"):
    model_name = os.path.normpath(os.path.basename(model_path))

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                trust_remote_code=True,
                quantization_config=bnb_config
            )

    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

    prompts = ["What is the capital of France? The capital is"]

    out = run_coupling_hf(model, tokenizer, model_name, prompts, save=True, verbose=False)

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", action="store", type=str)
    args = parser.parse_args()
    run(**vars(args))