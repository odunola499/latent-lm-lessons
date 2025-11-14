from queue import Queue
from threading import Thread
from latentlm.vae.audio.model import AcousticTokenizerModel, AcousticTokenizerConfig
from latentlm.vae.audio.cache import StreamingCache
import torchaudio
from torchaudio.io import StreamWriter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import time


def compute_receptive_field(kernel_sizes, dilation, strides):
    stride_product = 1
    receptive_field = 1
    for stride, kernel_size in zip(strides, kernel_sizes):
        receptive_field += (kernel_size - 1) * dilation * stride_product
        stride_product *= stride
    return receptive_field


def load_model(device, repo_id='odunola/vibevoice_vae_weights'):
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


def audio_stream(file_path, chunk_size):
    waveform, sample_rate = torchaudio.load(file_path)
    print("Starting streaming")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
    waveform = waveform.mean(0, keepdim=True).unsqueeze(0)
    for i in range(0, waveform.shape[-1], chunk_size):
        yield waveform[..., i:i + chunk_size]


def producer(input_queue: Queue, file_path: str, chunk_size=6000):
    for chunk in audio_stream(file_path, chunk_size):
        input_queue.put(chunk)
        time.sleep(0.05)
    input_queue.put(None)


def consumer(
        input_queue: Queue,
        output_queue: Queue,
        model: AcousticTokenizerModel,
        cache: StreamingCache,
        sample_indices: torch.Tensor,
        device
):
    while True:
        chunk = input_queue.get()
        if chunk is None:
            output_queue.put(None)
            break
        print(f"Input to model: {chunk.shape}")
        recon_chunk, _ = model(
            chunk.to(device),
            cache=cache,
            sample_indices=sample_indices,
            use_cache=True,
            debug=False
        )
        output_queue.put(recon_chunk)


def save_to_disk(stream_output_path: str, output_queue: Queue):
    writer = StreamWriter(stream_output_path)
    writer.add_audio_stream(sample_rate=24000, num_channels=1)
    writer.open()

    while True:
        chunk = output_queue.get()
        if chunk is None:
            print("Reached end")
            writer.close()
            break
        chunk = chunk[0].detach().cpu()
        chunk = chunk.squeeze(0).unsqueeze(-1)
        writer.write_audio_chunk(0, chunk)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strides = [1, 2, 2, 4, 5, 5, 8]
    kernel_sizes = [7] + [i * 2 for i in strides[1:]]
    dilation = 1

    receptive_field = compute_receptive_field(kernel_sizes, dilation, strides=strides)

    input_queue = Queue()
    output_queue = Queue()
    model, cache = load_model(device)
    sample_indices = torch.tensor([0], device=device)

    file_path = 'audio.wav'
    full_output_path = 'offline_recon.wav'
    stream_output_path = 'stream_recon.wav'

    chunk_size = int(receptive_field * 1.3)
    print(F"Receptive field: {receptive_field}, Chunk_size: {chunk_size}")

    producer_thread = Thread(target=producer, args=(
        input_queue, file_path, chunk_size)
                             )
    consumer_thread = Thread(target=consumer, args=(
        input_queue, output_queue, model, cache, sample_indices, device)
                             )
    save_thread = Thread(target=save_to_disk, args=(stream_output_path, output_queue))

    producer_thread.start()
    consumer_thread.start()
    save_thread.start()

    producer_thread.join()
    consumer_thread.join()
    save_thread.join()

    print('Finished Streaming inference')

    audio, sample_rate = torchaudio.load(file_path)
    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=24000)
    audio = audio.unsqueeze(0).to(device)
    recon, _ = model(audio)
    torchaudio.save(full_output_path, recon[0].detach().cpu(), sample_rate=24000)

    print('Finished Offline inference')

