from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/dsl_wiki_5gram.klm"


def DSLWiki5gram(cache_dir=None):
    """
    DSL and wikipedia corpus trained 5-gram model.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_wiki_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="070287617eacbbde79df2be34ac9615f",
                     cache_dir=cache_dir,
                     file_type="language_model")
