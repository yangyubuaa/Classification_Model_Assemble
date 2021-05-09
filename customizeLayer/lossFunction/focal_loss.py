import torch

class FocalLoss(ttorch.nn.Module):
    def __init__(self， weight=None):
        super(FocalLoss, self).__init__()
        # 每个类别所占权重，如无必要，设置为None
        self.class_weight = weight

    
    def forward(self, prd, trg):
        """
        prd:    shape:(batch_size, class_nums)
        trg:    shape:(batch_size)
        """
        loss = 0
        
