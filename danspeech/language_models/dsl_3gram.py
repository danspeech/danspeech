from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/Rasmusafj/models_development/raw/master/da_lm_3gram_pruned_0_5_30.klm"


def DSL3gram(cache_dir=None):
    """
    DSL 3-gram language model. This is the best performing for out test cases along with DSL 5-gram.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="dsl_3gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="33ca3e2a8db3a036af6d7ad85972dbb0",
                     cache_dir=cache_dir,
                     file_type="language_model")
