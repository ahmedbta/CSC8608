import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

def main():
    os.makedirs("TP3/outputs", exist_ok=True)

    text = (
        "Thanks for calling. I am sorry your order arrived damaged. "
        "I can offer a replacement or a refund. "
        "Please confirm your preferred option."
    )

    # Utilisation directe de silero TTS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, example_text = torch.hub.load('snakers4/silero-models', 'silero_tts', language='en', speaker='v3_en')
    t0 = time.time()
    audio = model.apply_tts(text=text, speaker='en_0', sample_rate=48000)
    t1 = time.time()
    elapsed_s = t1 - t0
    audio_dur_s = len(audio) / 48000
    rtf = elapsed_s / max(audio_dur_s, 1e-9)
    out_wav = "TP3/outputs/tts_reply.wav"
    sf.write(out_wav, audio, 48000)

    print("tts_model: silero_tts (en)")
    print("device:", device)
    print("audio_dur_s:", round(audio_dur_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_wav)

if __name__ == "__main__":
    main()
