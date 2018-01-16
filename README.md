# Neural Network Pruning PyTorch Implementation

Luyu Wang & Gavin Ding

Borealis AI

## Motivation
Neural network pruning has become a trendy research topic, but we haven't found an easy to use PyTorch implementation. We want to take advantage of the power of PyTorch and build pruned networks to study their properties.

**Note**: this implementation is not aiming at obtaining computational efficiency but to offer convenience for studying properties of pruned networks. Discussions on how to have an efficient implementation is welcome. Thanks!

## High-level idea
1. We write [wrappers](https://github.com/wanglouis49/pytorch-weights_pruning/blob/master/pruning/layers.py) on PyTorch Linear and Conv2d layers.
2. For each layer, once a binary mask tensor is computed, it is multiplied with the actual weights tensor on the forward pass.
3. Multiplying the mask is a differentiable operation and the backward pass is handed by automatic differentiation (no explicit coding needed).

## Pruning methods

### Weight pruning
Han et al propose to compress deep learning models via weights pruning [Han et al, NIPS 2015](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network). This repo is an implementation in PyTorch. The pruning method is replaced by the "class-blinded" method mentioned in [See et al, CoNLL 2016](https://arxiv.org/abs/1606.09274), which is much easier to implement and has better performance as well.

### Filter pruning
Pruning convolution filters has the advantage that it is more hardware friendly. We also implement the "minimum weight'' approach in [Molchanov et al, ICLR 2017](https://arxiv.org/abs/1611.06440)

