import torch
from torch import nn
from typing import Optional

class Cache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id:str):
        states = self.cache.get(layer_id, None)
        return states

    def set(self, layer_id:str, states:torch.Tensor):
        self.cache[layer_id]= states.detach()

    def clear(self):
        self.cache = {}

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, bias=bias)

        context_size = (kernel_size - 1) * dilation - (stride - 1)
        self.context_size = context_size if context_size > 0 else 0
        self.padding = self.context_size

    def forward(self, tensor, cache:Optional = None):
        if cache is not None:
            return self._forward_streaming(tensor, cache)
        return self._forward_offline(tensor)

    @property
    def layer_id(self):
        return str(id(self))

    def _forward_streaming(self, tensor, cache:Cache):
        B, C, T = tensor.shape
        cached_states = cache.get(self.layer_id)
        if cached_states is None:
            cached_states = torch.zeros(B, C, self.context_size, device = tensor.device, dtype = tensor.dtype)

        input_with_context = torch.cat([cached_states, tensor], dim = 2)
        output = self.conv(input_with_context)

        if self.context_size > 0:
            total_input_length = input_with_context.shape[2]
            if total_input_length >= self.context_size:
                new_cache_start = total_input_length - self.context_size
                new_cache = input_with_context[:, :, new_cache_start:]
            else:
                new_cache = input_with_context

            cache.set(self.layer_id, new_cache)
        return output

    def _forward_offline(self, tensor):
        tensor = nn.functional.pad(tensor,(self.padding, 0))
        return self.conv(tensor)

class ConvModule(nn.Module):
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

    def forward(self, tensor:torch.Tensor, cache:Optional[Cache] = None):
        output = tensor
        for layer in self.layers:
            output = layer(output, cache)
        return output

channels = [1,3,5,7,11]
kernel_sizes = [3,3,3,3]
dilations = [1,2,1,2]
strides = [2,1,2,1]

cache = Cache()
causal_conv_stack = ConvModule(
    channels=channels,
    kernel_sizes=kernel_sizes,
    dilations = dilations,
    strides = strides
)
test_audio = torch.randn(2,1,24000)
output_frames = []

chunk_size = 2000 # New chunk of audio stream that keeps coming in
for i in range(0, test_audio.shape[-1], chunk_size):
    chunk = test_audio[...,i:i+chunk_size]
    output_frame = causal_conv_stack(
        chunk, cache = cache
    )
    output_frames.append(output_frame)

print('streaming inference')
stream_output = torch.concat(output_frames, dim = -1)
print(stream_output.shape)

print('offline inference')
offline_output = causal_conv_stack(test_audio)
print(offline_output.shape)

print(torch.abs(stream_output-offline_output).mean())


