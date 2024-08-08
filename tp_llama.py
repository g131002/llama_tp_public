import os

import torch
import torch.distributed as dist
from tp_modeling_llama import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer
from torch.optim import Adam

def train_test():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    hf_token = os.environ['HF_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hf_token)

    test_input = 'test llama forward backward'
    inputs = tokenizer(test_input, return_tensors='pt').to(device)
    inputs['labels'] = inputs['input_ids']

    config = LlamaConfig(num_hidden_layers=1, hidden_size=1024, intermediate_size=3000)
    initial_llama_state_dict = torch.load('initial_llama.pth')
    llama = LlamaForCausalLM(config, initial_llama_state_dict).to(device)

    optimizer = Adam(llama.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
    loss = llama(**inputs).loss
    loss.backward()

    ref_grad = torch.load('embed_grad.pth').to(device)
    assert torch.allclose(ref_grad, llama.model.embed_tokens.weight.grad, atol=1e-2)

    optimizer.step()

    one_step_llama_state_dict = torch.load('one_step_llama.pth')

    with torch.no_grad():
        ref_state_dict = one_step_llama_state_dict
        model_state_dict = llama.state_dict()

        for k in ref_state_dict.keys():
            ref = ref_state_dict[k].float()
            current = model_state_dict[k].cpu().float()
            assert torch.allclose(
                ref, current, atol=1e-2
            ), f"Model state dict does not match the reference model state dict for key {k}. Difference: {(ref - current).abs().max()}"

        print("Model state dict matches the reference model state dict")

    print('test')


if __name__ == "__main__":
    train_test()