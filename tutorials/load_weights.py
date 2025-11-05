from huggingface_hub import hf_hub_download, list_repo_files, create_repo, HfApi
import json
from safetensors.torch import load_file, save_file
import torch
from pathlib import Path
from latentlm.vae.audio.model import SemanticTokenizerConfig, AcousticTokenizerConfig
from latentlm.vae.audio.model import SemanticTokenizerModel, AcousticTokenizerModel


repo_id = 'microsoft/VibeVoice-1.5B'
local_dir = Path('./checkpoints')

files = list_repo_files(repo_id)
shard_files = [local_dir / f for f in files if 'safetensor' in f and '.json' not in f]


for file in shard_files:
    print(f"Downloading {file}")
    hf_hub_download(
        repo_id = repo_id,
        filename = file.name,
        local_dir = local_dir,
        local_dir_use_symlinks=False
    )

acoustic_weights = {}
semantic_weights = {}

for shard in shard_files:
    shard_state_dict = load_file(shard)
    for param_name, param in shard_state_dict.items():
        if 'acoustic_tokenizer' in param_name:
            acoustic_weights[param_name] = param
        elif 'semantic_tokenizer' in param_name:
            semantic_weights[param_name] = param
        else:
            continue


semantic_config = SemanticTokenizerConfig()
acoustic_config = AcousticTokenizerConfig()

semantic_model = SemanticTokenizerModel(semantic_config)
acoustic_model = AcousticTokenizerModel(acoustic_config)

semantic_model.load_state_dict(semantic_weights)
acoustic_model.load_state_dict(acoustic_weights)

save_file(semantic_weights, 'semantic.safetensors')
save_file(acoustic_weights, 'acoustic.safetensors')

print('checkpoints loaded')

repo_id = "odunola/vibevoice_vae_weights"
create_repo(repo_id, exist_ok = True)


api = HfApi()
api.upload_file(
    path_or_fileobj="semantic.safetensors",
    path_in_repo="semantic.safetensors",
    repo_id=repo_id,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="acoustic.safetensors",
    path_in_repo="acoustic.safetensors",
    repo_id=repo_id,
    repo_type="model"
)
