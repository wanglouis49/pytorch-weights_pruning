import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         

def test(model, loader):

    print('Checking accuracy on test set')  
    model.eval()

    num_correct, num_samples = 0, len(loader)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % 
        (num_correct, num_samples, 100 * acc))
    return acc
    

def prune_rate(model):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    for i, parameter in enumerate(model.parameters()):

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            print("Layer {} | {} | {}% parameters pruned".format(
                i, 
                'Conv' if len(parameter.data.size()) == 4 else 'Linear',
                zero_param_this_layer/param_this_layer,
                ))

    print("Final pruning rate: ", nb_zero_param/total_nb_param)
    return nb_zero_param, total_nb_param


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    