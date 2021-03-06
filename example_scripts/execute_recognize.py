"""
Transcribing a single audio file
================================

In this example script, DanSpeech is used to transcribe the same audio file with three different outputs:

- **Greedy decoding**: using no external language model.

- **Beam search decoding 1**: Decoding with a language model (:meth:`language_models.DSL3gram`).

- **Beam search decoding 2**: Decoding with a language model (:meth:`language_models.DSL3gram`) and returning all the beam_width most probable beams.

"""

from danspeech import Recognizer
from danspeech.pretrained_models import TestModel
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio

# Load a DanSpeech model. If the model does not exists, it will be downloaded.
model = TestModel()
recognizer = Recognizer(model=model)

# Load the audio file.
audio = load_audio(path="../example_files/u0013002.wav")

print()
print("No language model:")
print(recognizer.recognize(audio))

# DanSpeech with a language model.
# Note: Requires ctcdecode to work!
try:
    lm = DSL3gram()
    recognizer.update_decoder(lm=lm, alpha=1.2, beta=0.15, beam_width=10)
except ImportError:
    print("ctcdecode not installed. Using greedy decoding.")

print()
print("Single transcription:")
print(recognizer.recognize(audio, show_all=False))

print()
beams = recognizer.recognize(audio, show_all=True)
print("Most likely beams:")
for beam in beams:
    print(beam)

