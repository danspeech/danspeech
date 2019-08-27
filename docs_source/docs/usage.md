# Usage

```python
import danspeech
from danspeech.pretrained_models import TestModel
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio

# Load a DanSpeech model. If the model does not exists, it will be downloaded. 
model = TestModel()
recognizer = danspeech.Recognizer(model=model)

# Load the audio file.

audio = load_audio(path="./example_files/u0013002.wav")
print(recognizer.recognize(audio))

# DanSpeech with a language model.
# Note: Requires ctcdecode to work! 
lm = DSL3gram()
recognizer.update_decoder(lm=lm, alpha=1.3, beta=0.15, beam_width=32)
print(recognizer.recognize(audio, show_all=False))
```