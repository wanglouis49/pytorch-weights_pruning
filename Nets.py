import torch.nn as nn
from Layers import *

    
# LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = MaskedConv2d(1, 20, kernel_size=5, padding=0, stride=1)
        self.conv2 = MaskedConv2d(20, 50, kernel_size=5, padding=0, stride=1)
        self.linear1 = MaskedLinear(4*4*50, 500)
        self.linear2 = MaskedLinear(500, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out

    def set_masks(self, masks):
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.linear1.set_mask(masks[2])
        self.linear2.set_mask(masks[3])

    
    
