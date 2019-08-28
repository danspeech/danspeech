"""
Stream of audio from your microphone
====================================

This is an example of using
"""

from danspeech import Recognizer
from danspeech.pretrained_models import DanSpeechPrimary
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio

# Load a DanSpeech model. If the model does not exists, it will be downloaded.
model = DanSpeechPrimary()
recognizer = Recognizer(model=model)

# Load the audio file.
audio = load_audio(path="../example_files/u0013002.wav")

print("No language model:")
print(recognizer.recognize(audio))

# DanSpeech with a language model.
# Note: Requires ctcdecode to work!
lm = DSL3gram()
recognizer.update_decoder(lm=lm, alpha=1.2, beta=0.15, beam_width=10)

print("Single transcription:")
print(recognizer.recognize(audio, show_all=False))


print("Most likely beams:")
print(recognizer.recognize(audio, show_all=True))