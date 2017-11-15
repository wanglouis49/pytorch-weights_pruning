import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler


def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):
    num_epochs = param['num_epochs']
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = to_var(x)
            y_var = to_var(y.long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         

def check_accuracy(model, loader):
    model.eval()
    print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    for x, y in loader:
        x_var = to_var(x, volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return acc
    

def check_nonzero(model):
    param_count = 0
    nonzero = 0
    for parameter in model.parameters():
        total_params = 1
        for dim in parameter.data.size():
            total_params *= dim
        param_count += total_params
        if len(parameter.data.size()) != 1:
            # only pruning linear and conv layers
            nonzero_params = np.count_nonzero(parameter.cpu().data.numpy())
        else: 
            nonzero_params = len(parameter.cpu().data.numpy())
        nonzero += nonzero_params
        print(parameter.data.size(), nonzero_params/total_params)
    print("Final pruning percentage: ", nonzero/param_count)
    return nonzero, param_count
    