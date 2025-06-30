import os
import torch
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# ---- CONFIGURATION ----
checkpoint_path = "./tts_output/best_model.pth"  # change if needed
config_path = "./config.json"
output_onnx_path = "./glow_tts_export.onnx"
# ------------------------

# Load config
config = GlowTTSConfig()
config.load_json(config_path)

# Initialize audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)
config.num_chars = tokenizer.characters.num_chars
config.encoder_hidden_channels = 192  # use your trained config value if different

# Build model
model = GlowTTS(config)
model.load_checkpoint(config, checkpoint_path, eval=True, ap=ap, tokenizer=tokenizer)
model.eval()

# Create dummy input tensors
input_text = torch.randint(0, config.num_chars, (1, 50), dtype=torch.long)
input_lengths = torch.tensor([50], dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    (input_text, input_lengths),
    output_onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input_text", "input_lengths"],
    output_names=["mel_output", "alignment", "duration"],
    dynamic_axes={"input_text": {1: "seq_len"}, "mel_output": {2: "mel_len"}},
)

print(f"✅ ONNX model exported to: {output_onnx_path}")
