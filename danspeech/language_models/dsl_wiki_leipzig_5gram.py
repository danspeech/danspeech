from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def DSLWikiLeipzig5gram(cache_dir=None):
    """
    DSL, wikipedia and Leipzig corpus trained 5-gram model.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_wiki_leipzig_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="toDO",
                     cache_dir=cache_dir,
                     file_type="language_model")
