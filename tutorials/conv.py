import torch
from torch import nn
import math

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


tensor = torch.randn(8, 128, 64)
kernel_size = 3
stride = 2

pads = get_extra_padding_for_conv1d(tensor, kernel_size, stride)
print(pads)

padded_tensor = nn.functional.pad(tensor, (1, pads))
conv = nn.Conv1d(128, 128, kernel_size, stride)
output = conv(tensor)
print(output.shape)

output = conv(padded_tensor)
print(output.shape)