import torch

def rbind_list(x):
    out = {name: torch.cat([torch.tensor(element[name]).unsqueeze(0) for element in x]) for name in x[0].keys()}
    return out
