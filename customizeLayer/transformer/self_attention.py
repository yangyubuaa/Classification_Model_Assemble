import torch
import sys
sys.path.append("../../")
from utils.load import load_yaml
from customizeLayer.functional.softmax import softmax_tensor

class SelfAttention(torch.nn.Module):
    def __init__(self, configs):
        super(SelfAttention, self).__init__()

        self.configs = configs
        self.d_model = self.configs["d_model"]

        self.Wq = torch.randn((self.d_model, self.d_model), requires_grad=True)
        self.Wk = torch.randn((self.d_model, self.d_model), requires_grad=True)
        self.Wv = torch.randn((self.d_model, self.d_model), requires_grad=True)

    
    def forward(self, input, attention_mask):
        # input shape(batch_size, max_seq_len, d_model)
        # attention_mask shape(batch_size, max_seq_len)
        q = torch.matmul(input, self.Wq)
        k = torch.matmul(input, self.Wk)
        v = torch.matmul(input, self.Wv)

        k = k.permute(0, 2, 1)

        print(q.shape)
        print(k.shape)
        attention_score = torch.zeros((12, 12)).unsqueeze(0)
        for i in range(q.shape[0]):
            sample_score = torch.matmul(q[i, :, :], k[i, :, :])
            sample_score = sample_score.unsqueeze(0)
            attention_score = torch.cat([attention_score, sample_score], 0)
        attention_score = attention_score[1:, :, :]

        # 将attention_mask引入
        mask_attention_score = torch.ones((input.shape[0], attention_score.shape[0], attention_score.shape[0]))  # (batch_size, 12, 12)
        mask_attention_score = mask_attention_score * 1000
        print(mask_attention_score)
        # 更新每个sample的attention score
        for i in range(input.shape[0]):
            # 获取每个sample的attention score
            for j in range(attention_mask[i]):
                for k in range(attention_mask[i]):
                    mask_attention_score[i][j][k] = attention_score[i][j][k]

        print(mask_attention_score)
        attention_score_n = softmax_tensor(attention_score)
        print(attention_score_n)




if __name__=="__main__":
    configs = load_yaml("/home/ubuntu1804/pytorch_sequence_classification/customizeLayer/transformer/transformer_encoder_config.yaml")
    s = SelfAttention(configs)
    input = torch.randn((10, 12, 512))
    attention_mask = torch.tensor([1, 4, 5, 2, 2, 8, 6, 4, 8, 4]).unsqueeze(-1)
    print(attention_mask)
    s(input, attention_mask)