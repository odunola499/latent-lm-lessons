from tqdm.auto import tqdm
from latentlm.vae.audio.model import AcousticTokenizerModel, AcousticTokenizerConfig
from latentlm.vae.audio.cache import StreamingCache
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

def load_model(device, repo_id = 'odunola/vibevoice_vae_weights'):
    cache = StreamingCache()
    config = AcousticTokenizerConfig()
    model = AcousticTokenizerModel(config).to(device)
    path = hf_hub_download(
        repo_id = repo_id,
        filename = 'acoustic.safetensors'
    )
    weights = load_file(path)
    model.load_state_dict(weights)
    return model, cache


class Audio:
    def __init__(self, audio_file:str, window_size = 0.2, stride = 0.2):
        audio, sr = torchaudio.load(audio_file)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
        audio = audio.unsqueeze(0)
        self.audio = audio
        self.max_length = self.audio.shape[-1]
        self.window_length = window_size * 24000
        self.stride_length = int(stride * 24000)
        self.current_pointer = 0

    def get_full_audio(self):
        return self.audio

    def __iter__(self):
        self.current_pointer = 0
        while self.current_pointer < self.max_length:
            start = self.current_pointer
            end = start + self.window_length
            window = self.audio[..., start:end]
            self.current_pointer += self.stride_length

            if window.shape[-1] < self.window_length:
                pad = self.window_length - window.shape[-1]
                window = torch.nn.functional.pad(window, (0, pad))
            yield window


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cache = load_model(device)
    model.eval()
    test_audio = 'audio.wav'
    audio = Audio(test_audio)
    full_audio = audio.get_full_audio()

    print("Running Offline Inference")
    with torch.no_grad():
        full_recon, _ = model(full_audio, cache=None, use_cache=False, debug = True)

    cache.clear()
    sample_indices = torch.tensor([0], device=device)
    streaming_chunks = []

    print('Running streaming Inference')
    iterator = iter(audio)
    with torch.no_grad():
        for chunk in iterator:
            recon_chunk, _ = model(
                chunk,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=True,
                debug=True
            )
            streaming_chunks.append(recon_chunk)

    streaming_recon = torch.cat(streaming_chunks, dim=-1)

    torchaudio.save('full_recon.wav', full_recon[0], sample_rate = 24000)
    torchaudio.save('stream_recon.wav', streaming_recon[0], sample_rate=24000)

    print('saved full recon audio')
    print('saved stream_recon_audio')