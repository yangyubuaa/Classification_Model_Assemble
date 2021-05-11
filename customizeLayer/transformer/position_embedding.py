import torch
import numpy as np
import sys
sys.path.append("../../")

from utils.load import load_yaml

class RelativePositionEmbedding:
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

if __name__=="__main__":
    configs = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/customizeLayer/transformer/transformer_encoder_config.yaml")
    p = RelativePositionEmbedding(configs)
    print(p.get_position_embedding())