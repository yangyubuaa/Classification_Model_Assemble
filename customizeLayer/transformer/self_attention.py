import torch
import math
import sys
sys.path.append("../../")
from utils.load import load_yaml
from customizeLayer.functional.softmax import softmax_tensor

class SelfAttention(torch.nn.Module):
    def __init__(self, configs):
        super(SelfAttention, self).__init__()

        self.configs = configs
        self.d_model = self.configs["d_model"]

        self.Wq = torch.nn.Parameter(torch.randn((self.d_model, self.d_model), requires_grad=True))
        self.Wk = torch.nn.Parameter(torch.randn((self.d_model, self.d_model), requires_grad=True))
        self.Wv = torch.nn.Parameter(torch.randn((self.d_model, self.d_model), requires_grad=True))

    
    def forward(self, input, attention_mask):
        # input shape(batch_size, max_seq_len, d_model)
        # attention_mask shape(batch_size, max_seq_len)
        q = torch.matmul(input, self.Wq)
        k = torch.matmul(input, self.Wk)
        v = torch.matmul(input, self.Wv)

        k = k.permute(0, 2, 1)

        # print(q)
        # print(k)
        attention_score = torch.zeros((input.shape[1], input.shape[1])).unsqueeze(0)
        for i in range(q.shape[0]):
            sample_score = torch.matmul(q[i, :, :], k[i, :, :])
            sample_score = sample_score.unsqueeze(0)
            attention_score = torch.cat([attention_score, sample_score], 0)
        attention_score = attention_score[1:, :, :]
        # print(attention_score)
        attention_score = attention_score / math.sqrt(self.d_model**3)

        print(attention_score.shape) # (bsz, max_seq_len, max_seq_len)

        # 将attention_mask引入
        mask_attention_score = torch.ones((input.shape[0], attention_score.shape[1], attention_score.shape[1]))  # (batch_size, max_seq_len, max_seq_len)
        mask_attention_score = mask_attention_score * -1000

        print(mask_attention_score.shape)
        # print(mask_attention_score)
        # 更新每个sample的attention score
        for i in range(input.shape[0]):
            # 获取每个sample的attention score
            for j in range(attention_mask[i]):
                for k in range(attention_mask[i]):
                    mask_attention_score[i][j][k] = attention_score[i][j][k]

        # print(mask_attention_score)
        attention_score_n = softmax_tensor(mask_attention_score)
        print(attention_score_n.shape) # (bsz, seq_len, seq_len)

        # 按照score进行合并
        # attention_score shape (bsz, seq_len, seq_len)
        # input shape (bsz, seq, d_model)

        self_attention = torch.matmul(attention_score, input)
        return self_attention



if __name__=="__main__":
    configs = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/customizeLayer/transformer/transformer_encoder_config.yaml")
    s = SelfAttention(configs)
    input = torch.randn((10, 50, 512))
    attention_mask = torch.tensor([1, 4, 5, 2, 2, 8, 6, 4, 8, 4]).unsqueeze(-1)
    # print(attention_mask)
    s(input, attention_mask)
    for name in s.state_dict():
        print(name)