import torch
import sys
sys.path.append("../../")

from transformers import XLNetModel

from utils.load import load_yaml

class XLNetClassification(torch.nn.Module):
    def __init__(self, configs):
        super(XLNetClassification, self).__init__()

        self.configs = configs

        self.xlnet_model_path = self.configs["model_path"]["xlnet_path"]
        self.xlnet_d_model = self.configs["model_params"]["xlnet_d_model"]
        self.label_nums = configs["model_params"]["label_nums"]
        self.linear_mid_dimension["model_params"]["linear_mid_dimension"]

        self.xlnet_model = XLNetModel.from_pretrained(self.xlnet_model_path)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.xlnet_d_model, self.linear_mid_dimension),
            nn.ReLU(),
            torch.nn.Linear(self.linear_mid_dimension, self.label_nums)
        )
        

    def forward(self, input):
        last_hidden_state = self.xlnet_model(input, return_dict=True)  # shape of last_hidden_state:(bsz, num_predict, hidden_size)
        classify_character = last_hidden_state[:, 0, :].squeeze()
        return self.classifier(classify_character)


if __name__=="__main__":
    # 打印网络各层的名称
    configs = load_yaml("xlnet_classification_config.yaml")
    x = XLNetClassification(configs)
    for name in x.state_dict():
        print(name)
