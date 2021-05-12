import torch
import sys
sys.path.append("../../../")

from customizeLayer.transformer.position_embedding import AbsPositionEmbedding
from utils.load import load_yaml


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, configs):
        super(TransformerEncoderLayer, self).__init__()

        self.configs = configs
        self.vocab_size = configs["vocab_size"]
        self.word_embedding_dim = self.configs["word_embed_dim"]

        self.word_embedding = torch.nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.position_embedding = AbsPositionEmbedding(self.configs)

    def forward(self, input, attention_mask):
        # shape of input (bsz, batch_seq_len)
        # shape of attention_mask (bsz, 1)
        word_embedded = self.word_embedding(input)

        position_embedded = self.position_embedding(input, attention_mask)
        
        embedding = word_embedded + position_embedded

        print(embedding.shape)
        return embedding

if __name__=="__main__":
    configs = load_yaml("transformer_encoder_configs.yaml")
    transformer_encoder_layer = TransformerEncoderLayer(configs)
    input = torch.tensor(
        [
            [1, 3, 12, 4, 134, 5, 2],
            [6, 3, 1, 4, 0, 0, 0],
            [8, 645, 3, 1, 4, 5, 0],
            [76, 3, 1, 7, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0]
        ]
    )
    attention_mask = torch.tensor([7, 4, 6, 4, 1]).unsqueeze(-1)
    transformer_encoder_layer(input, attention_mask)
    
