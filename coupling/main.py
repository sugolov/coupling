import os
from collections import defaultdict

import torch

from .jacobian import jacobian, svd
from .metrics import metrics
from .utils import timestamp

def coupling_from_hooks(hooks, p=2, num_sing_vecs=(10,30,50), index=-1, index_in=None, 
    activation=None, chunks=4, verbose=False, device="cuda"):
    """
    Computes the coupling of residual Jacobians across hooks.

    hooks:      dict of representations before and after skip connection
    - hooks[layer] = {0: x_in, 1: x_out}
    p:              order of p-norm for coupling measurement
    num_sing_vecs:  number of top singular vectors to use in computing coupling 
    index:          output token index for Jacobian
    index_in:       input token index for Jacobian   
    - by default uses `index`
    activation: specifies whether to apply activation to `x_out` before computing Jacobian
    chunks:     number of chunks in Jacobian computation
    """
    Jac = []

    for h in hooks:
        timestamp("computing J of: ", h) if verbose else None

        x_in = hooks[h][0]
        x_out = hooks[h][1]
        dim = x_out.shape[-1]

        if activation is None:
            J = jacobian(x_out, x_in, index=-1, device="cuda", chunks=chunks).detach()
            Jac.append(J - torch.eye(dim).to(device))
            timestamp("Jacobian shape ", J.shape) if verbose else None
        else:
            J = jacobian(activation(x_out), x_in, index=index, index_in=index_in, device="cuda").detach()
            Jac.append(J - torch.eye(dim).to(device))

    timestamp("Computing coupling metrics") if verbose else None

    Us, Ss, Vs = svd(Jac)
    coupling_ujv, coupling_vju = metrics(Jac, Us, Ss, Vs, p=p, num_sing_vecs=num_sing_vecs)

    return coupling_ujv, coupling_vju


def run_coupling_hf(model, tokenizer, model_name, prompts, start=None, end=None, \
        save=False, out_path=None, device="cuda", verbose=False):
    """
    Runs coupling experiment for HuggingFace model and tokenizer. (See `demo/minimal_demo_hf.py`)

    model:      model following HuggingFace API
    tokenizer:  tokenizer of model
    model_name: model name, e.g. `Meta-Llama-3-8B`
    promtpts:   list of str prompts to pass to tokenizer
    """
    out = defaultdict(dict)
    start = start if start is not None else 0
    end = end if end is not None else len(prompts)

    for i, prompt in zip(range(start, end), prompts):
        timestamp(f"Running prompt {i + 1} of {end}")

        out[i] = {"prompt": prompt}
        print(prompt) if verbose else None

        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        num_tokens = input_ids.shape[1]
        print("Number of tokens:", num_tokens) if verbose else None

        chunks = 2 * (num_tokens // 20) + 5 + i
        print("Number of chunks:", chunks) if verbose else None

        input_ids = input_ids.to('cuda')
        outputs = model(input_ids, output_hidden_states=True)
        L = len(outputs.hidden_states) - 1 

        # format as hooks
        outputs_zip = {}
        for j in range(L):
            outputs_zip[f"block_{j}"] = {0: outputs.hidden_states[j], 1: outputs.hidden_states[j+1]}

        # compute coupling
        coupling_ujv, coupling_vju = coupling_from_hooks(
            outputs_zip, activation=None, chunks=chunks, verbose=verbose, device=device
        )

        out[i]["coupling_ujv"] = coupling_ujv
        out[i]["coupling_vju"] = coupling_vju
        timestamp(f"Ended prompt") if verbose else None

    if save:
        if out_path is None:
            out_path = os.getcwd()
            print(f"Saving enabled but out_path path not specified. Saving in `{out_path}`")
        out_file = os.path.join(out_path, "_".join((model_name, "coupling.pt")))
        torch.save(out, out_file)

    return out