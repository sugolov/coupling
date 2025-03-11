import os
import torch

from align import jacobian
from metrics import metrics
from utils import timestamp

def coupling_from_hooks(hooks, p=2, activation=None, chunks=4, verbose=False):
    """
    Computes the coupling of residual Jacobians across hooks.
    """
    Jacobians = []

    for h in hooks:
        timestamp("computing J of: ", h) if verbose else None

        x_in = hooks[h][0]
        x_out = hooks[h][1]
        dim = x_out.shape[-1]

        if activation is None:
            #last token
            J = jacobian(x_out, x_in, -1, "cuda", chunks).detach()
            Jacobians.append(J - torch.eye(dim))

            timestamp("Jacobian shape ", J.shape) if verbose else None
        else:
            #last token
            J = jacobian(activation(x_out), x_in, -1, "cuda").detach()
            Jacobians.append(J - torch.eye(dim))

    timestamp("computing coupling metrics") if verbose else None

    aln_ujv_all_k, aln_vju_all_k, _, _, _, _ = metrics(Jacobians)   

    # TODO: aggregate all the metrics here
    return aln_ujv_all_k, aln_vju_all_k


def run_coupling(model, x_batch, save_dir=None):
    activations = {}
    out = {}

    def get_block_hook(block_idx: int):
        def hook(module, input, output):
            activations[f"block_{block_idx}"] = (input[0], output)
        return hook

    timestamp("creating hooks")
    # Register hooks
    handles = []
    for idx, block in enumerate(model.blocks):
        handles.append(
            block.register_forward_hook(get_block_hook(idx))
        )
    
    timestamp("starting forward pass")
    model(x_batch)
    timestamp("done")
    # print(activations)vit_training/deit/align.py
    timestamp("block 0 in shape: " + str(activations["block_0"][0].shape))
    timestamp("block 0 out shape: " + str(activations["block_0"][1].shape))
    dim = activations["block_0"][1].shape[-1]

    timestamp("computing Jacobians")
    Jac_noskip = []
    model.eval()
    for idx in activations:
        timestamp("computing " + idx)
        x_in = activations[idx][0]
        x_out = activations[idx][1]

        #last token
        J = jacobian(x_out, x_in, -1, "cuda").detach()
        timestamp("Jacobian shape ", J.shape)
        Jac_noskip.append(J - torch.eye(dim))

    timestamp("computing coupling metrics")
    aln_ujv_all_k, aln_vju_all_k, _, _, _, _ = alignment_metric_new(Jac_noskip)   

    # TODO: replace with coupling_from_hooks(hooks)
    
    out["aln_ujv_all_k"] = aln_ujv_all_k
    out["aln_vju_all_k"] = aln_vju_all_k

    return out

def run_coupling_hf(model, tokenizer, model_name, prompts, start=None, end=None, save=False, out_path=None):
    
    out = {}
    start = start if start is not None else 0
    end = end if end is not None else len(prompts)

    for i, prompt in zip(range(start, end), prompts):
        timestamp(f"Running prompt {i + 1} of {end}")
        out[i] = {"prompt": prompt}
        print(prompt)

        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        num_tokens = input_ids.shape[1]
        print(num_tokens)
        chunks = 2 * (num_tokens // 20) + 5 + i
        
        print("Number of chunks:", chunks)
        
        input_ids_cuda = input_ids.to('cuda')
        outputs = model(input_ids, output_hidden_states=True)
        L = len(outputs.hidden_states) - 1 # do not do last layer
        
        # zip to pass as hooks
        outputs_zip = {}
        for i in range(L):
            outputs_zip[f"block_{i}"] = {0: outputs.hidden_states[i], 1: outputs.hidden_states[i+1]}

        aln_ujv_all_k, aln_vju_all_k = coupling_from_hooks(outputs_zip, activation=None, chunks=chunks)

        out[i]["aln_ujv_all_k"] = aln_ujv_all_k
        out[i]["aln_vju_all_k"] = aln_vju_all_k

        # ujv_mat, vju_mat, uu_mat, vv_mat, uv_mat, vu_mat = compute_alignment_new(model, input_ids_cuda, chunks)
        
        # out[i]["alignment_uu"] = uu_mat
        # out[i]["alignment_vv"] = vv_mat
        # out[i]["alignment_uv"] = uv_mat
        # out[i]["alignment_vu"] = vu_mat
        # out[i]["alignment_ujv"] = ujv_mat
        # out[i]["alignment_vju"] = vju_mat

        timestamp(f"Ended prompt")

    if save:
        if out_path is None:
            out_path = os.getcwd()
            print(f"Saving enabled but out_path path not specified. Saving in {out_path}")
        out_file = os.path.join(out_path, "_".join(model_name, "coupling.pt"))
        torch.save(out, out_file)

    return out, out_file