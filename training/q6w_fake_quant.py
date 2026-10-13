import types

import torch
import torch.nn.functional as F
from torch import nn


QMIN = -31
QMAX = 31


def fake_quantize_q6w(weight):
    """Rowwise signed-int6 fake quantization with an identity STE."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Q6W fake quantization expects a floating-point matrix")
    maxima = weight.detach().float().abs().amax(dim=1, keepdim=True)
    scales = torch.where(maxima > 0, maxima / QMAX, torch.ones_like(maxima))
    quantized = torch.round(weight.float() / scales).clamp(QMIN, QMAX)
    dequantized = (quantized * scales).to(dtype=weight.dtype)
    return weight + (dequantized - weight).detach()


def _embedding_forward(module, input):
    weight = fake_quantize_q6w(module.weight)
    return F.embedding(
        input,
        weight,
        module.padding_idx,
        module.max_norm,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
    )


def _linear_forward(module, input):
    return F.linear(input, fake_quantize_q6w(module.weight), module.bias)


def install_q6w_fake_quant(model):
    """Patch embedding and linear forwards without changing parameters/state_dict."""
    if getattr(model, "_q6w_fake_quant_installed", False):
        raise RuntimeError("Q6W fake quantization is already installed")
    embeddings = 0
    linears = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            if module.weight.ndim != 2 or module.weight.shape[1] % 4:
                raise ValueError("Embedding shape is incompatible with the packed Q6W format")
            module.forward = types.MethodType(_embedding_forward, module)
            embeddings += 1
        elif isinstance(module, nn.Linear):
            if module.weight.ndim != 2 or module.weight.shape[1] % 4:
                raise ValueError("Linear shape is incompatible with the packed Q6W format")
            module.forward = types.MethodType(_linear_forward, module)
            linears += 1
    if embeddings != 1 or linears != 36:
        raise ValueError(f"Expected 1 embedding and 36 linear layers, found {embeddings} and {linears}")
    model._q6w_fake_quant_installed = True
    return embeddings, linears
