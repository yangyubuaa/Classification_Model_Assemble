import torch
import numpy as np


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma, weight=None):
        super(FocalLoss, self).__init__()

        # focal loss调节参数
        # 减少模型过于关注更新不好的样本，好与不好的样本会缩放相同的比例，过于大的导数缩放的多，如果模型过于关注预测不好的样本，那么将此值调小
        self.alpha = alpha
        # gamma越大，对预测不好的样本的导数值就越大，模型越关注预测不好的样本，预测好的样本的导数会变小
        self.gamma = gamma
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
            sample_loss = -self.alpha*(1-torch.log(optimize_trg))**self.gamma*torch.log(optimize_trg)
            batch_loss = batch_loss + sample_loss

        return batch_loss

if __name__=="__main__":
    f = FocalLoss(alpha=1, gamma=2)

    prd = torch.tensor([[0.2, 0.4, 0.4],
                        [0.1, 0.4, 0.5]], requires_grad=True)

    trg = torch.tensor([1, 0]).unsqueeze(-1)
    loss = f(prd, trg)
    print(loss)
    loss.backward()
    print(prd.grad)

    # 验证
    # print(np.log(2.5) + np.log(10))