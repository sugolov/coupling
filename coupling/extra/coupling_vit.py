
import torch
from metrics import metrics

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