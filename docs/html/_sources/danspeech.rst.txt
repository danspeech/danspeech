==================
What is DanSpeech?
==================

DanSpeech is an open-source Danish speech recognition (speech-to-text) python package based on the
`PyTorch <https://pytorch.org/>`_ deep learning framework. It was developed as part of a Master's thesis at DTU compute
by Martin Carsten Nielsen and Rasmus Arpe Fogh Jensen, supervised by Professor Lars Kai Hansen.

All DanSpeech models are end-to-end `DeepSpeech 2 <https://arxiv.org/abs/1512.02595>`_ models, trained on danish data with a CTC loss. The models are trained with various data agumentations to multiply the rather small amount of public speech recognition
data available in Danish.

The models may be combined with a language model through `beam-search decoding <https://arxiv.org/pdf/1408.2873.pdf>`_
to achieve the best results, DanSpeech provides language models trained on a large danish corpus as part of the released package.

While DanSpeech models perform state-of-the-art speech recognition in Danish, performance is
not perfect, and results are conditioned on specific use-cases.

Danspeech provides:

- An easy-to-use Recognizer that supports different use-cases for Danish speech recognition.
- Pre-trained models of varying sizes and complexities.
- Pre-trained language models.

Motivation
----------
We believe that speech recognition in Danish should be freely available for everyone to use. Therefore we decided to develop
an open-source, and easy-to-use automatic speech recognition system for Danish.

We believe that an open-source solution can play an important role in ensuring that Danish speech recognition
systems are not continually out-shined by English systems.

Speech recognition will inevitably be a big part of future IT innovations. And without an easy-to-use, and free system, innovative spirits
with a desire to utilize speech recognition in the development of Danish technologies might be hindered by cost barriers.

As such DanSpeech can be used commercially by companies without the resources to develop their own speech recognition
systems, or companies who simply do not wish to outsource this part of their pipeline. Deploying DanSpeech models instead of using an external API will, in addition to reducing costs, also reduce latency drastically, if deployed locally with a GPU.

Performance
-----------
We benchmarked the system on two Danish benchmarks, namely the publicly available
`Nordisk Språkteknologi <https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en>`_ (NST)
dataset and our own DanSpeech dataset (~1000 noisy recordings).
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
To test the DanSpeech models on your own data (both pre-recorded and streaming audio is supported), we have created a demo
that runs as a development django server on localhost. It is easy to install and hence easy to test the models
with a GUI (little technical knowledge is required to play around with the demo).

The demo also features a demo of a DanSpeech model adopted/finetuned to transcribe meetings from Folketinget
(The Danish Parliament), which demonstrates the power of finetuning models to specific domains.

For more info of the DanSpeech demo, see :ref:`demo`.

Train or Finetune DanSpeech models
----------------------------------
If you require better performance than what is apparent from the DanSpeech pre-trained models, we've also created
a github repository (`danspeech_training <https://github.com/danspeech/danspeech_training>`_), where you can train completely new models from scratch or finetune existing DanSpeech models
to your specific domain/use-case (recommended method).

Training new, or finetuning, DanSpeech models is useful if you have specific knowledge about the domain you wish
to apply Danish speech recognition to, **and** you have either domain specific text resources or, in the best case, speech data available.

Finetuning a DanSpeech model can result in much better performance but does require a certain level of technical expertise and
a GPU for training. As an example of performance for such a system, see :ref:`demo`.

For more information, see :ref:`training-repo`.
