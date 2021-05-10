import torch
import numpy as np


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(FocalLoss, self).__init__()
        # 每个类别所占权重，如无必要，设置为None
        self.class_weight = weight

    
    def forward(self, prd, trg):
        """
        prd:    shape:(batch_size, class_nums)
        trg:    shape:(batch_size)
        """
        batch_size = prd.shape[0]

        batch_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        for predict_result_index in range(batch_size):
            # shape(label_nums)
            predict_result = prd[predict_result_index]
            # shape(1)
            label = trg[predict_result_index]
            optimize_trg = predict_result[label]
            sample_loss = -torch.log(optimize_trg)
            batch_loss = batch_loss + sample_loss

        return batch_loss

if __name__=="__main__":
    f = FocalLoss()

    prd = torch.tensor([[0.2, 0.4, 0.4],
                        [0.1, 0.4, 0.5]], requires_grad=True)

    trg = torch.tensor([1, 0]).unsqueeze(-1)
    loss = f(prd, trg)
    print(loss)
    loss.backward()
    print(prd.grad)

    # 验证
    # print(np.log(2.5) + np.log(10))