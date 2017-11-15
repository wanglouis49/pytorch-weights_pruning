import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from Nets import *
from Utils import *
from Layers import *


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128, 
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.MNIST(root='../data/', train=True, download=True, transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=param['batch_size'])


# Load the pretrained model
net = LeNet5()
net.load_state_dict(torch.load('models/lenet5_pretrained.pkl'))
if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
net.train()


# prune the weights
masks = class_blinded_prune(net, param)
net.set_masks(masks)


# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])
train(net, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
acc = check_accuracy(net, loader_test)
check_nonzero(net)


# Save and load the entire model
torch.save(net.state_dict(), 'models/lenet5_pruned_'+prune_perc+'.pkl')
