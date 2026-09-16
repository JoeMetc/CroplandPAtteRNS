#####
# Author: Joseph Metcalfe
# Creation Date: 11/08/2025
# Purpose: Wrapper to enforce standardisation of loss function params
#####


import torch.nn as nn


class CEMaskedLoss(nn.Module):

    def __init__(self, config, maskBackground):
        super().__init__()
        self.config = config

        self.MaskedClass = 0 if maskBackground else 19 # 19 hits 'void' for PASTIS and isn't present in MTLCC's ids

        self.ce_loss = nn.CrossEntropyLoss(weight=None, ignore_index=self.MaskedClass, label_smoothing=0.025)

    def forward(self, preds, labels):

        ce = self.ce_loss(preds, labels)

        return ce




