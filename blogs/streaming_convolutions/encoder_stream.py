from queue import Queue
from threading import Thread
from latentlm.vae.audio.model import AcousticTokenizerModel, AcousticTokenizerConfig
from latentlm.vae.audio.model import SemanticTokenizerModel, SemanticTokenizerConfig
from latentlm.vae.audio.cache import StreamingCache
import torchaudio
from torchaudio.io import StreamWriter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import time
from typing import Union
import math


def pad_to_multiple(x: torch.Tensor, multiple: int = 3200) -> torch.Tensor:
    L = x.shape[-1]
    target_len = multiple * math.ceil(L / multiple)

    if L < target_len:
        pad_amount = target_len - L
        x = torch.nn.functional.pad(x, (0, pad_amount))

    return x


def compute_receptive_field(kernel_sizes, dilation, strides):
    stride_product = 1
    receptive_field = 1
    for stride, kernel_size in zip(strides, kernel_sizes):
        receptive_field += (kernel_size - 1) * dilation * stride_product
        stride_product *= stride
    return receptive_field


def load_acoustic_model(device, repo_id='odunola/vibevoice_vae_weights'):
    cache = StreamingCache()
    config = AcousticTokenizerConfig()
    model = AcousticTokenizerModel(config).to(device)
    path = hf_hub_download(
        repo_id=repo_id,
        filename='acoustic.safetensors'
    )
    weights = load_file(path)
    model.load_state_dict(weights)
    return model, cache


def load_semantic_model(device, repo_id='odunola/vibevoice_vae_weights'):
    cache = StreamingCache()
    config = SemanticTokenizerConfig()
    model = SemanticTokenizerModel(config).to(device)
    path = hf_hub_download(
        repo_id=repo_id,
        filename='semantic.safetensors'
    )
    weights = load_file(path)
    model.load_state_dict(weights)
    return model, cache


def audio_stream(file_path, chunk_size):
    waveform, sample_rate = torchaudio.load(file_path)
    print("Starting streaming")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
    waveform = waveform.mean(0, keepdim=True).unsqueeze(0)
    for i in range(0, waveform.shape[-1], chunk_size):
        yield waveform[..., i:i + chunk_size]


def producer(input_queue: Queue, file_path: str, chunk_size=6000):
    for chunk in audio_stream(file_path, chunk_size):
        chunk = pad_to_multiple(chunk, chunk_size)
        input_queue.put(chunk)
        time.sleep(0.05)
    input_queue.put(None)


def consumer(
        input_queue: Queue,
        output_queue: Queue,
        model: SemanticTokenizerModel,
        cache: StreamingCache,
        sample_indices: torch.Tensor,
        device
):
    while True:
        chunk = input_queue.get()
        if chunk is None:
            output_queue.put(None)
            break
        print(f"input shape: {chunk.shape}")
        _, latents = model(
            chunk.to(device), cache=cache, sample_indices=sample_indices, use_cache=True
        )
        print(f"Output latent shape {latents.shape}")
        output_queue.put(latents)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strides = [1, 2, 2, 4, 5, 5, 8]
    kernel_sizes = [7] + [i * 2 for i in strides[1:]]
    dilation = 1

    receptive_field = compute_receptive_field(kernel_sizes, dilation, strides=strides)

    input_queue = Queue()
    output_queue = Queue()
    model, cache = load_semantic_model(device)
    sample_indices = torch.tensor([0], device=device)

    file_path = 'audio.wav'
    chunk_size = 6400
    print(F"Receptive field: {receptive_field}, Chunk_size: {chunk_size}")

    producer_thread = Thread(target=producer, args=(
        input_queue, file_path, chunk_size)
                             )
    consumer_thread = Thread(target=consumer, args=(
        input_queue, output_queue, model, cache, sample_indices, device)
                             )
    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    collected = []

    while True:
        item = output_queue.get()
        if item is None:
            break
        collected.append(item.detach())

    stream_results = torch.concat(collected, dim=1)

    streamed_latents = torch.concat(collected, dim=1)
    print("Streaming complete")

    audio, sample_rate = torchaudio.load(file_path)
    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=24000)
    audio = audio.unsqueeze(0).to(device)
    audio = pad_to_multiple(audio, chunk_size)
    cache = StreamingCache()
    _, latents = model(audio)
    print('offline complete')

    print(f'streamed latents: {streamed_latents.shape}')
    print(f"offline latents: {latents.shape}")
    print(torch.abs(streamed_latents - latents).mean())
