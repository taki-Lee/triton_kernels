import json
import torch

def load_bgmv_inputs(path):
    with open(path, 'r') as f:
        info = json.load(f)
    print(info.keys())
    # print(info.values())
    input_embs = torch.tensor(info['input_embs'], device='cuda')
    embed_dim_ = info['base_layer_infer.embed_dim_']
    key_buffer = torch.tensor(info['self.key_buffer[layer_id]'], device='cuda')
    a_start = torch.tensor(info['self.infer_adapter.a_start'], device='cuda')
    a_len = torch.tensor(info['self.infer_adapter.a_len'], device='cuda')
    a_loc = torch.tensor(info['self.infer_adapter.a_loc'], device='cuda')
    a_scaling = torch.tensor(info['self.infer_adapter.a_scaling'], device='cuda')
    req_bins = torch.tensor(info['self.req_bins'], device='cuda')

    return (input_embs.view(-1, embed_dim_), key_buffer, 
            a_start, a_len, a_loc, a_scaling, req_bins)

