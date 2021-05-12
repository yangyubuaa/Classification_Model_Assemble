import torch

class LayerNormalization(torch.nn.Module):
    def __init__(self, configs):
        super(LayerNormalization, self).__init__()
        self.configs = configs

        self.normalized_dim = self.configs["layer_normalized_dim"]
        self.eps = self.configs["layer_normalized_eps"]

        self.alpha = torch.nn.Parameter(torch.ones(self.normalized_dim))
        self.beta = torch.nn.Parameter(torch.zeros(self.normalized_dim))
    
    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        
        return self.alpha * (input - mean) / (std + self.eps) + self.beta


if __name__=="__main__":
    configs = {
        "layer_normalized_dim": 512,
        "layer_normalized_eps": 512
    }
    l = LayerNormalization(configs)

    input = torch.randn((10, 12, 512))
    print(l(input).shape)
