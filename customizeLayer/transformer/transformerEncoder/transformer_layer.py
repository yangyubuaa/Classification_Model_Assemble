import torch

import sys
sys.path.append("../../../")

from customizeLayer.transformer.base.self_attention import SelfAttention

class TransformerLayer(torch.nn.Module):
    def __init__(self, configs):
        super(TransformerLayer, self).__init__()
        self.configs = configs

        self.atten_Layer = SelfAttention(configs)

    def forward(self, input, attention_mask):
        attention_ = self.atten_Layer(input, attention_mask)
        print(attention_.shape)