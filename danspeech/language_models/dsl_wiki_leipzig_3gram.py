from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/dsl_wiki_leipzig_3gram.klm"


def DSLWikiLeipzig3gram(cache_dir=None):
    """
    DSL, wikipedia and Leipzig corpus trained 3-gram model.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_wiki_leipzig_3gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="8409a469be718209afdd18692a2d5609",
                     cache_dir=cache_dir,
                     file_type="language_model")
