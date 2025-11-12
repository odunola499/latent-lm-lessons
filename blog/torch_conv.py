import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)

input = torch.randn(1,3,10)
causal_conv = CausalConv1d(3,5, kernel_size = 3, dilation = 2)
output = causal_conv(input)
print(output.shape)
# Size([1,5,10])



class ConvModule(nn.Module):
    def __init__(self, channels:list, kernel_sizes:list, dilations:list, strides:list, bias = False):
        super().__init__()
        num_layers = len(channels) - 1
        assert len(kernel_sizes) == len(dilations) == len(strides) == num_layers
        layers = [
            nn.Conv1d(
                in_channels = channels[i],
                out_channels = channels[i+1],
                kernel_size = kernel_sizes[i],
                stride = strides[i],
                dilation = dilations[i],
                bias = bias,
            )
        for i in range(num_layers)]
        self.layers = nn.Sequential(*layers)

        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.strides = strides

    def forward(self, tensor):
        output = tensor
        for layer in self.layers:
            output = layer(output)
        return output

    def get_receptive_field(self):
        stride_product = 1
        receptive_field = 0
        for kernel_size, dilation, stride in zip(self.kernel_sizes, self.dilations, self.strides):
            receptive_field += (kernel_size - 1) * dilation * stride_product
            stride_product *= stride
        return receptive_field


channels = [1,3,5,7,11]
kernel_sizes = [3,3,3,3]
dilations = [1,2,1,2]
strides = [2,1,2,1]

conv_stack = ConvModule(
    channels=channels,
    kernel_sizes=kernel_sizes,
    dilations = dilations,
    strides = strides
)
# input_signal = torch.randn(2,1,24000)
# output = conv_stack(input_signal)
# print(output.shape)
# print(conv_stack.get_receptive_field())

class CausalConvModule(nn.Module):
    def __init__(self, channels:list, kernel_sizes:list, dilations:list, strides:list, bias = False):
        super().__init__()
        num_layers = len(channels) - 1
        assert len(kernel_sizes) == len(dilations) == len(strides) == num_layers
        layers = [
            CausalConv1d(
                in_channels = channels[i],
                out_channels = channels[i+1],
                kernel_size = kernel_sizes[i],
                stride = strides[i],
                dilation = dilations[i],
                bias = bias,
            )
        for i in range(num_layers)]
        self.layers = nn.Sequential(*layers)

        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.strides = strides

    def forward(self, tensor):
        output = tensor
        for layer in self.layers:
            output = layer(output)
        return output

    def get_receptive_field(self):
        stride_product = 1
        receptive_field = 0
        for kernel_size, dilation, stride in zip(self.kernel_sizes, self.dilations, self.strides):
            receptive_field += (kernel_size - 1) * dilation * stride_product
            stride_product *= stride
        return receptive_field

channels = [1,3,5,7,11]
kernel_sizes = [3,3,3,3]
dilations = [1,2,1,2]
strides = [2,1,2,1]

causal_conv_stack = ConvModule(
    channels=channels,
    kernel_sizes=kernel_sizes,
    dilations = dilations,
    strides = strides
)
input_signal = torch.randn(2,1,24000)
output = causal_conv_stack(input_signal)
print(output.shape)
print(conv_stack.get_receptive_field())