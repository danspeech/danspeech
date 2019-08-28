.. _pre-trained-models:

============================
Pre-trained DanSpeech Models
============================

All of the available DanSpeech models are shown below. If you need to finetune or train your own model,
then you can find more information at ???? LINK

Recommended usage for all models (except a custom model):

.. code-block:: python

    from danspeech.pretrained_models import TestModel
    model = TestModel()


Available models
----------------
.. automodule:: danspeech.pretrained_models
    :members: DanSpeechPrimary, TestModel, Baseline, TransferLearned, Folketinget, EnglishLibrispeech,
        CPUStreamingRNN, GPUStreamingRNN, CustomModel