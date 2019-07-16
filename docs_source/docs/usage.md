# Usage

```python
import danspeech
from danspeech.pretrained_models import Units400
from danspeech.language_models import DSL3gram
from danspeech.audio.resources import SpeechFile

# Load a DanSpeech model. If the model does not exists, it will be downloaded. 
model = Units400()
recognizer = danspeech.Recognizer(model=model)

# Load the speech file.
with SpeechFile(filepath="./example_files/u0013002.wav") as source:
    audio = recognizer.record(source)
   
print(recognizer.recognize(audio))

# DanSpeech with a language model.
# Note: Requires ctcdecode to work! 
lm = DSL3gram()
recognizer.update_decoder(lm=lm, alpha=1.3, beta=0.15, beam_width=32)
print(recognizer.recognize(audio, show_all=True))

```