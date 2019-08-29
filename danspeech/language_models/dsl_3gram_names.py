from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/dsl_names.klm"


def DSL3gramWithNames(cache_dir=None):
    """
    Includes DSL + a bias towards the most common names in Denmark.

    DSL 3-gram language model. This is the best performing for out test cases along with DSL 5-gram.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_names.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="1b47e2db841c6be5c62004ef51a40c68",
                     cache_dir=cache_dir,
                     file_type="language_model")
