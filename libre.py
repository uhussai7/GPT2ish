import torchaudio
dataset = torchaudio.datasets.LIBRISPEECH(root="./data", url="test-clean", download=True)
waveform, sample_rate, transcript, *_ = dataset[0]
torchaudio.save("sample_audio.wav", waveform, sample_rate)