# DanSpeech
An open-source python package for Danish speech recognition.


# Installation
Current setup is 

```bash
pip install . -r requirements.txt 
```

If you require beam CTC decoding, then you additional need to
install [ctcdecode](https://github.com/parlance/ctcdecode).

# Usage

```python
import danspeech
from danspeech.pretrained_models import Units400
from danspeech.language_models import DSL3gram
from danspeech.audio.resources import SpeechFile

# Load a DanSpeech model. If the model does not exists, it will be downloaded. 
model = Units400()
recognizer = danspeech.Recognizer(model=model)

# Load the speech file
with SpeechFile(filepath="./example_files/u0013002.wav") as source:
    audio = recognizer.record(source)
   
print(recognizer.recognize(audio))

# DanSpeech with a language model.
# Note: Requires ctcdecode to work! 
lm = DSL3gram()
recognizer.update_decoder(lm=lm, alpha=1.3, beta=0.15, beam_width=32)
print(recognizer.recognize(audio, show_all=True))

```

# Demo
To experience the models, we've created a demo. The demo resides at [https://github.com/rasmusafj/danspeechdemo](https://github.com/rasmusafj/danspeechdemo).

# Support
If you require help with the software, then feel free to create issues here on github. We will continually solve issues
and answer any questions that you might have. 


# Authors and acknowledgment
Main authors: 
* Martin Carsten Nielsen  ([mcnielsen4270@gmail.com](mcnielsen4270@gmail.com))
* Rasmus Arpe Fogh Jensen ([rasmus.arpe@gmail.com](rasmus.arpe@gmail.com))

Other acknowledgements:

* We've trained the models based on the code from [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
* The audio handling and recognizing flow is based on [https://github.com/Uberi/speech_recognition](https://github.com/Uberi/speech_recognition)
* Handling of the pretrained models is based on [keras](https://github.com/keras-team/keras)
* We've trained all models with the aid of DTU using data from

# Project status
The project is currently under development. We will soon give first release and we expect a stable release to ocurr
in late august 2019. If you want to contribute, then you are welcome to fork and create pull requests. 