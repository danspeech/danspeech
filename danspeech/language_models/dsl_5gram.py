from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/dsl_5gram.klm"


def DSL5gram(cache_dir=None):
    """
    DSL 5-gram language model. This is the best performing for out test cases along with DSL 3-gram.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="f2929d6d154b57b8be0c05347036c7e6",
                     cache_dir=cache_dir,
                     file_type="language_model")
