import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

def transcribe_audio_wav2vec(file_path):
    # Load tokenizer and model
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load and resample audio
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Tokenize and predict
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = tokenizer.decode(predicted_ids[0])
    return transcription.lower()

# === Example Usage ===
if _name_ == "_main_":
    # ✅ Use raw string or double backslashes to avoid path errors
    path = r"C:\Users\91911\Downloads\Record (online-voice-recorder.com).wav"
    print("🔊 Transcription:\n", transcribe_audio_wav2vec(path))
