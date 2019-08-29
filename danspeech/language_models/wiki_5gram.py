from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/wiki_5gram.klm"


def Wiki5gram(cache_dir=None):
    """
    wikipedia corpus trained 5-gram model.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="wiki_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="b329e215b2fde5ffe3e2c94204f6c189",
                     cache_dir=cache_dir,
                     file_type="language_model")
