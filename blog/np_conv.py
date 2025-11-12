import numpy as np

num_channels = 3
input_length = 10
kernel_size = 3
stride = 2


def conv1d(input:np.ndarray, kernel_size:int):
    num_channels, input_length = input.shape
    kernel = np.random.randn(num_channels, kernel_size)
    output_length = input_length - kernel_size + 1

    output = np.empty((num_channels, output_length), dtype = input.dtype)

    for channel in range(num_channels):
        for position in range(output_length):
            start = position
            end = start + kernel_size
            output[channel, position] = np.dot(kernel[channel,:], input[channel, start:end])

    return output

input = np.random.randn(num_channels, input_length)
output = conv1d(input, kernel_size)
print(output.shape)


num_channels = 3
input_length = 10
kernel_size = 3
stride = 2


def conv1d_with_stride(input:np.ndarray, kernel_size:int, stride:int):
    num_channels, input_length = input.shape
    kernel = np.random.randn(num_channels, kernel_size)
    output_length = ((input_length - kernel_size) // stride) + 1

    output = np.empty((num_channels, output_length), dtype = input.dtype)
    for channel in range(num_channels):
        for position in range(output_length):
            start = position * stride
            end = start + kernel_size
            output[channel, position] = np.dot(kernel[channel, :], input[channel, start:end])
    return output

input = np.random.randn(num_channels, input_length)
output = conv1d_with_stride(input, kernel_size, stride)
print(output.shape)


num_channels = 3
input_length = 10
kernel_size = 3
stride = 2
dilation = 2


def conv1d_with_stride_and_dilation(input:np.ndarray, kernel_size:int, stride:int, dilation:int):
    num_channels, input_length = input.shape
    kernel = np.random.randn(num_channels, kernel_size)

    effective_kernel = (kernel_size-1) * dilation + 1
    output_length = ((input_length - effective_kernel) // stride) + 1

    output = np.empty((num_channels, output_length), dtype = input.dtype)
    for channel in range(num_channels):
        for position in range(output_length):
            start = position * stride
            end = start + effective_kernel
            indices = np.arange(start, end, dilation)
            output[channel, position] = np.dot(kernel[channel, :], input[channel, indices])
    return output

input = np.random.randn(num_channels, input_length)
output = conv1d_with_stride_and_dilation(input, kernel_size, stride, dilation)
print(output.shape)

num_channels = 3
input_length = 10
kernel_size = 3
stride = 2
dilation = 2
padding = (2,2)


def conv1d_with_stride_dilation_padding(input:np.ndarray, kernel_size:int, stride:int, dilation:int, padding:tuple):
    num_channels, input_length = input.shape
    kernel = np.random.randn(num_channels, kernel_size)

    effective_kernel = (kernel_size-1) * dilation + 1
    output_length = ((input_length - effective_kernel + sum(padding)) // stride) + 1

    output = np.empty((num_channels, output_length), dtype = input.dtype)
    padded_input = np.pad(input, pad_width = ((0,0), padding), mode = 'constant', constant_values=0)

    for channel in range(num_channels):
        for position in range(output_length):
            start = position * stride
            end = start + effective_kernel
            indices = np.arange(start, end, dilation)
            output[channel, position] = np.dot(kernel[channel, :], padded_input[channel, indices])
    return output

input = np.random.randn(num_channels, input_length)
output = conv1d_with_stride_dilation_padding(input, kernel_size, stride, dilation, padding)
print(output.shape)

num_input_channels = 3
num_output_channels = 5
input_length = 10
kernel_size = 3
stride = 2
dilation = 2
padding = (2,2)

num_input_channels = 3
num_output_channels = 5
input_length = 10
kernel_size = 3
stride = 2
dilation = 2
padding = (2,2)

def general_conv1d(
        input:np.ndarray,
        num_output_channels,
        kernel_size:int,
        stride:int,
        dilation:int,
        padding:tuple,
        bias = True
):
    num_input_channels, input_length = input.shape
    kernel = np.random.randn(num_output_channels, num_input_channels,kernel_size)
    bias = np.random.randn(num_output_channels) if bias is True else np.zeros(num_input_channels)

    effective_kernel = (kernel_size-1) * dilation + 1
    output_length = ((input_length - effective_kernel + sum(padding)) // stride) + 1

    output = np.empty((num_output_channels,output_length), dtype = input.dtype)
    padded_input = np.pad(input, pad_width = ((0,0), padding), mode = 'constant', constant_values=0)

    for channel in range(num_output_channels):
        for position in range(output_length):
            start = position * stride
            end = start + effective_kernel
            indices = np.arange(start, end, dilation)
            value = 0
            for in_channel in range(num_input_channels):
                result = np.dot(kernel[channel, in_channel,:], padded_input[in_channel, indices])
                value += result
            output[channel, position] = value + bias[channel]
    return output

input = np.random.randn(num_input_channels, input_length)
output = general_conv1d(input, num_output_channels,kernel_size, stride, dilation, padding, bias = True)
print(output.shape)



import torch
from torch import nn

num_input_channels = 3
num_output_channels = 5
input_length = 10
kernel_size = 3
stride = 2
dilation = 2
padding = (2)
bias = True

conv = nn.Conv1d(
    in_channels = num_input_channels,
    out_channels = num_output_channels,
    kernel_size = kernel_size,
    stride = stride,
    dilation = dilation,
    padding = padding,
    bias = bias
)

input = torch.randn(num_input_channels,input_length)
output = conv(input)
print(output.shape)


num_input_channels = 3
num_output_channels = 5
input_length = 10
kernel_size = 3
stride = 2
dilation = 2

def causal_conv1d(
        input:np.ndarray,
        num_output_channels,
        kernel_size:int,
        stride:int,
        dilation:int,
        bias = True
):
    num_input_channels, input_length = input.shape
    kernel = np.random.randn(num_output_channels, num_input_channels,kernel_size)
    bias = np.random.randn(num_output_channels) if bias is True else np.ones(num_input_channels)

    effective_kernel = (kernel_size-1) * dilation + 1
    left_padding = (kernel_size - 1) * dilation
    padding = (left_padding, 0)

    output_length = ((input_length - effective_kernel + sum(padding)) // stride) + 1
    output = np.empty((num_output_channels,output_length), dtype = input.dtype)

    padded_input = np.pad(input, pad_width = ((0,0), padding), mode = 'constant', constant_values=0)

    for channel in range(num_output_channels):
        for position in range(output_length):
            start = position * stride
            end = start + effective_kernel
            indices = np.arange(start, end, dilation)
            value = 0
            for in_channel in range(num_input_channels):
                result = np.dot(kernel[channel, in_channel,:], padded_input[in_channel, indices])
                value += result
            output[channel, position] = value + bias[channel]
    return output

input = np.random.randn(num_input_channels, input_length)
output = causal_conv1d(input, num_output_channels,kernel_size, stride, dilation, bias = True)
print(output.shape)



