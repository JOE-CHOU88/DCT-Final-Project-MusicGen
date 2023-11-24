import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

print("-------Start Model Loaded-------")
try:
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
except Exception as e:
    print("error:",e)
print("-------Done Model Loaded-------")
model.set_generation_params(duration=10)  # generate 8 seconds.

descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
melody, sr = torchaudio.load('./assets/bach.mp3')

print("-------Start Stylizing Music-------")
# generates using the melody from the given audio and the provided descriptions.
try:
    wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
except Exception as e:
    print("error:",e)
print("-------Done Stylizing Music-------")

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

print("-------Finish-------")