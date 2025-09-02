import torch, torch.nn.functional as F
def top2_routing(logits):
    routing_weights = F.softmax(logits, dim=-1)
    return torch.topk(routing_weights, 2, dim=-1)
