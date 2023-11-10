from baseModel import half_unet
from torch import nn
from torch.autograd import Variable
import torch

import math


class featureExctra (nn.Module):
    def __init__(self,dropout = 0.1,hidden_dim = 512):
        super(featureExctra,self).__init__()

        self.h_unet = half_unet()

        self.position_embedding = PositionalEncoding(hidden_dim=hidden_dim,dropout=dropout)
