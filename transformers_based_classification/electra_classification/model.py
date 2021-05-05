import torch
import torch.nn as nn

from transformers import ElectraModel


class ElectraClassification(nn.Module):
    def __init__(self, configs):
        super(ElectraClassification, self).__init__()
        self.configs = configs
        self.bert_hiddensize = self.configs["bert_hiddensize"]
        self.dense = self.configs["dense"]
        self.label_nums = self.configs["label_nums"]
        self.dropout = self.configs["dropout"]

        self.electra_model = ElectraModel.from_pretrained(self.configs["path"]["electra_path"])
        # for p in self.bert_model.parameters():
        #     p.requires_grad = True
        # output shape of bert: (batch_size, seqlens, lstm_hiddensize)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(self.bert_hiddensize, self.dense),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dense, self.label_nums)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        electra_outputs = self.electra_model(input_ids, token_type_ids, attention_mask)
        last_hidden_state = electra_outputs.last_hidden_state #(bsz, seqlens, hiddensize)
        last_hidden_state = last_hidden_state[:, 0, :].squeeze()
        return self.classification(last_hidden_state)


if __name__ == '__main__':
    pass