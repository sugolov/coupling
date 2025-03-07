import torch

from align import jacobian
from metrics import metrics
from utils import timestamp
"""

"""

def coupling_from_hooks(hooks, p=2, activation=None, verbose=False):
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
            J = jacobian(x_out, x_in, -1, "cuda").detach()
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



#class ExperimentAlignNew:
#
#    def __init__(self,
#                models,
#                model_types,
#                tokenizer,
#                model_name,
#                out_dir="out",
#                alignment           = True,
#                linearize           = False
#                ):
"""

:param model:
:param tokenizer:

:param model_name:      name of model for outputs
:param out_dir:         output directory

:param lss:             True/False to include/exclude
:param equidistance:    True/False to include/exclude
:param norm:            True/False to include/exclude
:param head_entropy:    True/False to include/exclude

# were included:
self.models          = models
self.model_types     = model_types
self.tokenizer       = tokenizer
self.model_name      = model_name
self.out_dir         = out_dir

self.alignment       = alignment
self.linearize       = linearize
"""

def run_coupling_hf(self, dataset, data_name, start=None, end=None, save=True):

    if data_name == 'gsm8k' and end == 5:
        skip_4 = True
        end = 6
    else:
        skip_4 = False

    prompts, start, end = get_prompts(dataset, data_name, start=start, end=end)

    tokenizer = self.tokenizer
    model_types = self.model_types
    models = self.models


    outputs = {
        "DATA_NAME": data_name
    }

    out = {}


    for i, prompt in zip(range(start, end), prompts):
        if skip_4 and i == 4:
            continue
        timestamp(f"Running prompt {i + 1} of {end}")
        out[i] = {}
        print(prompt)

        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        num_tokens = input_ids.shape[1]
        print(num_tokens)
        chunks = 2 * (num_tokens // 20) + 5 + i
        
        print("Number of chunks:", chunks)# timestamp(f"Tokens loaded")

        for idx in range(len(models)):
            model_type = model_types[idx]
            model = models[idx]
            out[i][model_type] = {}
            

            if self.alignment:
                input_ids_cuda = input_ids.to('cuda')
                ujv_mat, vju_mat, uu_mat, vv_mat, uv_mat, vu_mat = compute_alignment_new(model, input_ids_cuda, chunks)
                
                out[i][model_type]["alignment_uu"] = uu_mat
                out[i][model_type]["alignment_vv"] = vv_mat
                out[i][model_type]["alignment_uv"] = uv_mat
                out[i][model_type]["alignment_vu"] = vu_mat
                out[i][model_type]["alignment_ujv"] = ujv_mat
                out[i][model_type]["alignment_vju"] = vju_mat
            # timestamp("Metric collection complete")
            #print(self.linearize)

            if self.linearize:
                cos, t = compute_linearization(model, input_ids_cuda, chunks)
                out[i][model_type]["cos_linearized"] = cos
                out[i][model_type]["t"] = t

            timestamp(f"Ended {model_type}")

    outputs["OUT"] = out

    if save:
        out_file = self._format_out_dir(data_name, model_type, start=start, end=end)
        torch.save(outputs, out_file)

    return outputs, out_file

def _format_out_dir(self, data_name, model_type, start=None, end=None):
    out_file = data_name + "_" + model_type + "_" + str(start) + "_" + str(end) + ".pt"

    if self.alignment:
        out_file = 'alignment_' + out_file

    experiment_out_dir = os.path.join(self.out_dir, self.model_name, data_name)

    if not os.path.exists(os.path.join(self.out_dir, self.model_name)):
        os.mkdir(os.path.join(self.out_dir, self.model_name))

    if not os.path.exists(experiment_out_dir):
        os.mkdir(experiment_out_dir)

    out_file = os.path.join(experiment_out_dir, out_file)

    return out_file