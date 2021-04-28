class ClassificationLabelTokenizer:
    def __init__(self, label2index: dict):
        if isinstance(label2index, dict):
            self.label2index = label2index
        else:
            self.label2index = load_json(label2index)
        self.index2label = {value:key for key, value in self.label2index.items()}

    def __call__(self, input_label, r_tensor=True):
        if isinstance(input_label, list):
            num_label = [self.label2index[i] for i in input_label]
            if r_tensor:
                return torch.tensor(num_label).unsqueeze(-1)
            else:
                return num_label
        else:
            num_label = self.label2index[input_label]
            if r_tensor:
                return torch.tensor(num_label).unsqueeze(-1)
            else:
                return num_label

    def decode(self, label_tensor):
        label_list = label_tensor.squeeze().numpy().tolist()
        if isinstance(label_list, int):
            return [self.index2label[label_list]]
        else:
            for index in range(len(label_list)):
                label_list[index] = self.index2label[label_list[index]]

            return label_list

if __name__ == '__main__':
    # ClassificationLabelTokenizer使用方法1
    classificationlabeltokenizer = ClassificationLabelTokenizer(
        "/Users/yangyu/PycharmProjects/infer_of_intent/data_preprocess/label2index.json")
    tokenized = classificationlabeltokenizer(["['护理_疾病护理']", "['护理_疾病护理']"])
    print(tokenized)