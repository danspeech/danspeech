# DanSpeech
An open-source python package for Danish speech recognition.

You can find all relevant information in the documentation and we provide you with some extra links below. 

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg?style=for-the-badge)](https://danspeech.github.io/danspeech/)

[![Installation](https://img.shields.io/badge/Installation-blue.svg?style=for-the-badge)](https://danspeech.github.io/danspeech/html/installation.html)

[![Examples](https://img.shields.io/badge/Examples-blue.svg?style=for-the-badge)](https://danspeech.github.io/danspeech/html/auto_examples/index.html)

## Demo
To experience the danspeech package, we've created a simple demo with a nice GUI. It depends on danspeech 
and django.

The demo resides at [https://github.com/danspeech/danspeechdemo](https://github.com/danspeech/danspeechdemo).

## Train models
If you wish to train your own models, or perhaps finetune a DanSpeech model to your specific use case, we then 
refer to the [DanSpeech training repository](https://github.com/danspeech/danspeech_training). 

## Support
If you require help with the software, then feel free to create issues here on github. We will continually solve issues
and answer any questions that you might have. 

## Authors and acknowledgment
Main authors: 
* Martin Carsten Nielsen  ([mcnielsen4270@gmail.com](mcnielsen4270@gmail.com))
* Rasmus Arpe Fogh Jensen ([rasmus.arpe@gmail.com](rasmus.arpe@gmail.com))

This project is supported by Innovation Foundation Denmark through the projects DABAI and ATEL

Other acknowledgements:

* We've trained the models based on the code from [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
* The audio handling and recognizing flow is based on [https://github.com/Uberi/speech_recognition](https://github.com/Uberi/speech_recognition).
* Handling of the pretrained models is based on [keras](https://github.com/keras-team/keras).
* We've trained all models with the aid of DTU using data from Sprakbanken ([NST](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en)).

## Licence
The software is in general licenced under the Apache-2 licence and may be used commercially. 

[![Licence](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/danspeech/LICENCE.txt)

**NOTE:**
Some code parts in DanSpeech contain links to other original sources. If this is the case, then the specific code 
part is licenced differently depending on the source and if you wish to redistribute DanSpeech code, then you must
make sure you also comply with the original code licences.  

The flac binaries are licenced under GPLv2. See more information at [Flac licence](https://github.com/Uberi/speech_recognition/blob/master/LICENSE-FLAC.txt).
