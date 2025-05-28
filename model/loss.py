import torch
from torch import nn

class CrossEntropyMultiRank(nn.Module):
    def __init__(self, device, tax_ranks, class_weights=None):
        super(CrossEntropyMultiRank, self).__init__()
        self.loss = dict()
        for tax_rank in tax_ranks:
            if class_weights is not None:
                weight = torch.Tensor(list(class_weights[tax_rank].values())).to(device)
            else:
                weight = None
                
            self.loss[tax_rank] = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, preds, labels, tax_rank):
        loss = self.loss[tax_rank](preds, labels)
        return loss
    