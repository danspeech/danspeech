.. _language-models:

===============
Language Models
===============
All of the available DanSpeech language models are shown below. If you need to finetune or train your own model,
then you can find more information at ???? LINK

Recommended usage for all language models (except a custom model):

.. code-block:: python

    from danspeech.language_models import DSL3gram
    lm = DSL3gram()


Method
------

All language models are n-gram models with modified Kneser-Ney smoothing constructed from large text-corpora.

They have been generated with the use of `kenLM <https://kheafield.com/code/kenlm/>`_.


Available models
----------------

.. automodule:: danspeech.language_models
    :members: DSL3gram, DSL5gram, DSL3gramWithNames, DSLWiki3gram, DSLWiki5gram, DSLWikiLeipzig3gram,
        DSLWikiLeipzig5gram, Wiki3gram, Wiki5gram, Folketinget3gram, CustomLanguageModel
