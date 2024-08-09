import os
import random

import numpy as np
import torch
import torch.distributed as dist
from modeling_llama import RefLlamaForCausalLM
from tp_modeling_llama import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer
from torch.optim import Adam

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_test():
    set_seed(42)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    hf_token = os.environ['HF_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hf_token)

    test_input = 'test llama forward backward'
    inputs = tokenizer(test_input, return_tensors='pt').to(device)
    inputs['labels'] = inputs['input_ids']

    config = LlamaConfig(num_hidden_layers=1, hidden_size=1024, intermediate_size=3000)
    ref_llama = RefLlamaForCausalLM(config).to(device)
    llama = LlamaForCausalLM(config, ref_llama.state_dict()).to(device)
    ref_optimizer = Adam(ref_llama.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
    ref_loss = ref_llama(**inputs).loss
    optimizer = Adam(llama.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
    loss = llama(**inputs).loss

    ref_loss.backward()
    loss.backward()

    # ref_grad = ref_llama.model.layers[0].self_attn.q_proj.weight.grad.split(llama.model.layers[0].self_attn.num_heads * llama.model.layers[0].self_attn.head_dim, dim=0)[rank]
    # assert torch.allclose(ref_grad, llama.model.layers[0].self_attn.q_proj.weight.grad, atol=1e-2)

    ref_optimizer.step()
    optimizer.step()

    with torch.no_grad():
        tp = dist.get_world_size()
        ref_state_dict = ref_llama.state_dict()
        model_state_dict = llama.state_dict()

        for k in ref_state_dict.keys():
            ref = ref_state_dict[k].cpu().float()
            current = model_state_dict[k].cpu().float()
            if k == 'model.layers.0.self_attn.q_proj.weight':
                ref = ref.split(llama.model.layers[0].self_attn.num_heads * llama.model.layers[0].self_attn.head_dim, dim=0)[rank]
            elif k == 'model.layers.0.self_attn.k_proj.weight':
                ref = ref.split(llama.model.layers[0].self_attn.num_key_value_heads * llama.model.layers[0].self_attn.head_dim, dim=0)[rank]
            elif k == 'model.layers.0.self_attn.v_proj.weight':
                ref = ref.split(llama.model.layers[0].self_attn.num_key_value_heads * llama.model.layers[0].self_attn.head_dim, dim=0)[rank]
            elif k == 'model.layers.0.self_attn.o_proj.weight':
                ref = ref.split(llama.model.layers[0].self_attn.hidden_size // tp, dim=1)[rank]
            elif k == 'model.layers.0.mlp.gate_proj.weight':
                ref = ref.split(llama.model.layers[0].mlp.intermediate_size, dim=0)[rank]
            elif k == 'model.layers.0.mlp.up_proj.weight':
                ref = ref.split(llama.model.layers[0].mlp.intermediate_size, dim=0)[rank]
            elif k == 'model.layers.0.mlp.down_proj.weight':
                ref = ref.split(llama.model.layers[0].mlp.intermediate_size, dim=1)[rank]
            
            assert torch.allclose(
                ref, current, atol=1e-2
            ), f"Model state dict does not match the reference model state dict for key {k}. Difference: {(ref - current).abs().max()}"

        print("Model state dict matches the reference model state dict")

    print('test')


if __name__ == "__main__":
    train_test()