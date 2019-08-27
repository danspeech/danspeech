# Recognizer

The DanSpeech Recognizer is a collection of speech recognition tools that work with 
pre-trained DanSpeech models. 

To use a Recognizer instance, you need to supply it with a [pre-trained DanSpeech model]((/docs/pre-trained-models#pre-trained-danspeech-models)).

## Example
```python

from danspeech import Recognizer
from danspeech.pretrained_models import TestModel
from danspeech.audio import load_audio

model = TestModel()
r = Recognizer(model=model)
audio = load_audio(path="./example_files/u0013002.wav")

print(r.recognize(audio))
```

## Functions


