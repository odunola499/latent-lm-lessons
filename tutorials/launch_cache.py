import torch
from latentlm.vae.audio.cache import StreamingCache
from latentlm.vae.audio.conv import SConv1d

cache = StreamingCache()
conv = SConv1d(
    128, 128, 3, 1, causal = True, pad_mode = 'causal', norm = 'none'
)
print(conv.layer_id)

batch_size = 8
input = torch.randn(batch_size, 128, 18)
sample_indices = torch.arange(batch_size)
layer_id = conv.layer_id

print(cache.get(layer_id, sample_indices))

output = conv._forward_streaming(input, cache, sample_indices, debug = True)
print(output)
print(cache.get(layer_id, sample_indices))

