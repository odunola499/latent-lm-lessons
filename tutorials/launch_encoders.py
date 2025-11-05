from latentlm.vae.audio.model import SemanticTokenizerConfig, AcousticTokenizerConfig
from latentlm.vae.audio.model import SemanticTokenizerModel, AcousticTokenizerModel
import torch
from torch import nn

semantic_config = SemanticTokenizerConfig()
acoustic_config = AcousticTokenizerConfig()

semantic_model = SemanticTokenizerModel(semantic_config)
acoustic_model = AcousticTokenizerModel(acoustic_config)

semantic_params = [p.numel() for p in semantic_model.parameters()]
acoustic_params = [p.numel() for p in acoustic_model.parameters()]

print(f"Num of Semantic_params: {sum(semantic_params)}")
print(f"Num of acoustic params: {sum(acoustic_params)}")

tensor = torch.randn((1,1,24000))
print(tensor.shape)

output = semantic_model(tensor, debug = True)
print(output)
