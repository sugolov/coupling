# coupling
### ([Project page](https://sugolov.github.io/coupling/)) ([arXiv](https://arxiv.org/abs/2407.07810)) ([ICLR 2025](https://iclr.cc/virtual/2025/poster/28555))
Code release for *Transformer Block Coupling and its Correlation with Generalization in LLMs* in ICLR2025.

#### Contributors
- [Murdock Aubry](https://murdockaubry.com/)
- [Haoming Meng](https://www.linkedin.com/in/haoming-meng-264870180/)
- [Anton Sugolov](https://sugolov.github.io)
- [Vardan Papyan](https://sites.google.com/view/vardan-papyan/home)

![alt text](.readme/image.png)

## Setup

### Using `pip`
```
pip install git+https://github.com/sugolov/coupling.git
```

### Using `git`
```
git clone https://github.com/sugolov/coupling.git
cd coupling
pip install -e .
```

## Demo
Minimal HuggingFace demo
```
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from coupling import coupling_from_hooks, run_coupling_hf

model_path = "meta-llama/Meta-Llama-3-8B"
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

out = run_coupling_hf(model, tokenizer, model_name, prompts, save=True, verbose=True)
```

#### Google Colab: https://colab.research.google.com/drive/1ronRmxr0yJO8Re0iJeqp055IoiU7oLOI?usp=sharing

#### Minimal HuggingFace Demo: `python coupling/demo/minimal_demo_hf.py`

## Description

A package for computing the transformer block coupling metric $m(J_1, J_2)$ from PyTorch hooks.
Demo code and notebooks included in `coupling/demo`.

For token embeddings across depth, represented by $X^{l+1} = F_{\text{block}}^{l+1} (X^{l})= X^l + f^{l+1} (X^l)$, we compute
1. The Jacobians
$$J_{t_1t_2}^l = \frac{\partial }{\partial x_{t_1}^{l-1}}\left( f^l (X^{l-1})\right)_{t_2}  \in \mathbb{R}^{d\times d}$$
2. For $J_1, J_2$, their singular value decompositions
$$J_1 = U_1S_1V_1^T \qquad J_2 = U_2S_2V_2^T$$
3. The coupling between the singular vectors, **normalized** by the $p$-norm of $S_1$, given by
$$m_K(J_1, J_2) = \frac{|| U_{2,K}^TJ_1 V_{2,K} - S_{1,K} ||_F}{\lVert s_{1, K}\rVert_p} = \frac{|| U_{2,K}^TU_1 S_1 V_1^T V_{2,K} - S_{1,K} ||_F}{\lVert s_{1, K}\rVert_p}$$
