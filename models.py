import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d 


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
    

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7*7*64, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))
