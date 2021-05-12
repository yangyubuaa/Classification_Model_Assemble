import torch
import numpy as np
import sys
sys.path.append("../../")

from utils.load import load_yaml

class RelativePositionEmbedding:
    """实现相对位置编码（直接获取编码，不可学习）
    """
    def __init__(self, configs):
        self.configs = configs
        self.embedding_dim = self.configs["position_embedding_dim"]
        self.max_seq_len = self.configs["max_seq_len"]
    
    def get_position_embedding(self):
        positional_encoding = np.array([
        [np.sin(pos / np.power(10000, 2 * i / self.embedding_dim)) if i%2==0 else 
         np.cos(pos / np.power(10000, 2*i/ self.embedding_dim))
         for i in range(self.embedding_dim) ]
         for pos in range(self.max_seq_len)])
   
        return torch.from_numpy(positional_encoding)


class AbsPositionEmbedding(torch.nn.Module):
    """实现绝对位置编码（参数可学习）
    """
    def __init__(self, configs):
        super(AbsPositionEmbedding, self).__init__()
        self.configs = configs
        self.max_position = self.configs["max_position"]
        self.embedding_dim = self.configs["abs_pos_embed_dim"]

        self.embedding = torch.nn.Embedding(self.max_position, self.embedding_dim)

    def forward(self, input, attention_mask):
        # shape of attention_mask: (bsz, 1), input_seq必须小于max_position
        bsz = attention_mask.shape[0]
        max_seq_len = input.shape[1]
        pos_init = torch.zeros((bsz, max_seq_len))
        for i in range(len(attention_mask)):
            for j in range(0, attention_mask[i][0].item()):
                pos_init[i][j] = j + 1 

        return self.embedding(pos_init.long())

if __name__=="__main__":
    configs = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/customizeLayer/transformer/transformer_encoder_config.yaml")
    p = RelativePositionEmbedding(configs)
    print(p.get_position_embedding())

    configs = {
        "max_position":512, 
        "abs_pos_embed_dim":512
    }
    a = AbsPositionEmbedding(configs)
    input = torch.randn(10, 12)
    attention_mask = torch.tensor([1, 4, 5, 2, 2, 8, 6, 4, 8, 4]).unsqueeze(-1)
    print(a(input, attention_mask).shape)