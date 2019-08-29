==================
What is DanSpeech?
==================

DanSpeech is an open-source Danish speech recognition (speech-to-text) python package based on the
`PyTorch <https://pytorch.org/>`_ deep learning framework. It was developed as part of a Master's thesis at DTU
by Martin Carsten Nielsen and Rasmus Arpe Figh Jensen with supervisor Professor Lars Kai Hansen.

All of the DanSpeech models are end-to-end `DeepSpeech 2 <https://arxiv.org/abs/1512.02595>`_ models ctc trained on danish
text with various data agumentations as an attempt multiply the rather small and public speech recognition
data available in Danish.

The models may further be combined with a language model through `beam-search decoding <https://arxiv.org/pdf/1408.2873.pdf>`_
for the best possible speech recognition.

The models perform state-of-the-art speech recognition in Danish but performance is
evidently not perfect and conditioned on specific use-cases.

Danspeech contains:

- An easy-to-use Recognizer that supports different use-cases for Danish speech recognition.
- Pre-trained models.
- Pre-trained language models.

Motivation
----------
We believe that speech recognition in Danish should be available for anyone. We therefore developed
an open-source and easy-to-use speech recognition system for Danish.

An open-source speech recognition system is important to ensure that Danish speech recognition performance
does not continue to fall miles behind the performance of English systems. Speech recognition will inevitable
be a big part of future IT innovations. Without an easy-to-use and free system, innovation in Danish technologies
utilizing speech recognition is hindered.

DanSpeech can be used commercially for companies without the resources to develop their own speech recognition
systems or companies who do not wish to outsource speech recognition. Deploying DanSpeech models instead of
using an external (expensive) API will furthermore reduce latency drastically, if deployed with a GPU.

The system can also be used as part of education at various universities in Denmark.


Performance
-----------
We benchmarked the system on two Danish benchmarks, namely the public available
`Nordisk Språkteknologi <https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en>`_ (NST)
dataset and our own (not public due to data gathering restrictions and GDPR) DanSpeech dataset (~1000 noisy recordings).
The performance is evaulated in `Word Error Rate <https://en.wikipedia.org/wiki/Word_error_rate>`_ (WER).


+-------------------+---------------------------------------------+-------------+
| Dataset           | Models                                      | Performance |
+===================+=============================================+=============+
| NST test          | DanSpeechPrimary + DSL5Gram (not pruned) LM | 12.85% WER  |
+-------------------+---------------------------------------------+-------------+
| DanSpeech dataset | TransferLearned + DSL5Gram (not pruned) LM  | 25.75% WER  |
+-------------------+---------------------------------------------+-------------+


DanSpeech Demo
--------------
To test the DanSpeech models on your own audio files or on your own speech, we additionally created a demo
that runs as a development django server on localhost. It is easy to install and hence easy to test the models
with a GUI (little technical knowledge is required to play around with the demo).

The demo also features a demo of a DanSpeech model adopted/finetuned to transcribe meetings from Folketinget
(The Danish Parliament) and the near perfect transcriptions achieved.

For more info of the DanSpeech demo, see :ref:`demo`.

Train or Finetune DanSpeech models
----------------------------------
If you require better performance than what is apparent from the DanSpeech pre-trained models, we've also created
a github repository, where you can train completely new models from scratch or finetune existing DanSpeech models
to your specific domain/use-case (preferred method).

For more information, see :ref:`training-repo`.

Training new models or finetuning DanSpeech models is useful if you have knowledge about the domain, which you wish
to apply Danish speech recognition to, **and** you have either text resources or in the best case speech data available.

Finetuning a DanSpeech model can result in much better performance but does require technical expertise and
an available GPU for training. As an example of performance for such a system, see :ref:`demo`.
