import torch

import sys
sys.path.append("../../../")

from customizeLayer.transformer.self_attention import SelfAttention

class TransformerLayer(torch.nn.Module):
    def __init__(self, configs):
        super(TransformerLayer, self).__init__()
        