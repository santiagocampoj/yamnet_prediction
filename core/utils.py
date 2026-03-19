import soundfile as sf
import numpy as np
import resampy


def load_audio_file(path, target_sr=16000):
    # exactamente lo mismo que inference.py pero como función
    wav_data, sr = sf.read(path, dtype=np.int16)
    waveform = (wav_data / 32768.0).astype('float32')
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != target_sr:
        waveform = resampy.resample(waveform, sr, target_sr)
    return waveform

def mean_embedding(waveform, yamnet_model):
    # exactamente scores, embeddings del inference.py
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy().mean(axis=0)

def extract_embedding(waveform, yamnet_model):
    # devuelve todos los frames, no solo la media
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()