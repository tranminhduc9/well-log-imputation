"""Exact affine fusion of parallel depthwise temporal branches.

This changes the parameterization of an already trained network, not its
training history. BatchNorm and all nonlinear blocks remain unchanged.
"""
from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F


def fold_temporal_branches(network):
    """Return an independent copy with one depthwise max-kernel convolution."""
    encoder = network.encoder
    branches, fusion = encoder.temporal_branches, encoder.temporal_fusion
    if branches is None or fusion is None:
        raise ValueError("The network has no parallel temporal branches to fold.")
    channels = encoder.total_channels
    kernels = [layer.kernel_size[0] for layer in branches]
    largest = max(kernels)
    if any(layer.groups != channels or layer.in_channels != channels
           or layer.out_channels != channels or layer.stride != (1,)
           or layer.dilation != (1,) or layer.padding != (kernel//2,)
           or kernel % 2 == 0 for layer,kernel in zip(branches,kernels)):
        raise ValueError("Only centered, stride-one depthwise branches are supported.")
    if (fusion.groups != channels or fusion.kernel_size != (1,)
            or fusion.in_channels != channels*len(branches)
            or fusion.out_channels != channels):
        raise ValueError("Fusion must combine scales independently for each latent channel.")
    reference = branches[0].weight
    # Construction should not change the caller's random stream. The sampled
    # initialization is immediately replaced by the analytically folded values.
    with torch.random.fork_rng(devices=[]):
        temporal = nn.Conv1d(channels,channels,largest,padding=largest//2,
                             groups=channels,bias=True).to(reference)
    with torch.no_grad():
        weights = torch.zeros_like(temporal.weight)
        bias = torch.zeros_like(temporal.bias)
        if fusion.bias is not None:
            bias.copy_(fusion.bias)
        coefficients = fusion.weight[:,:,0]
        for scale,(layer,kernel) in enumerate(zip(branches,kernels)):
            padded = F.pad(layer.weight,((largest-kernel)//2,)*2)
            weights.add_(padded*coefficients[:,scale,None,None])
            if layer.bias is not None:
                bias.add_(coefficients[:,scale]*layer.bias)
        temporal.weight.copy_(weights)
        temporal.bias.copy_(bias)
    folded=deepcopy(network)
    folded.encoder.temporal=temporal
    folded.encoder.temporal_branches=None
    folded.encoder.temporal_fusion=None
    folded.config=replace(network.config,multi_scale_conv=False,kernel_size=largest)
    folded.train(network.training)
    return folded
