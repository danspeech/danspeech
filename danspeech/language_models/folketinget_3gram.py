from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/da_lm_3gram_folketinget.klm"


def Folketinget3gram(cache_dir=None):
    """
    3-gram language model trained on all meeting summaries from the Danish Parliament (Folketinget)

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/lms/`` folder.
    :return: path to .klm language model
    :rtype: str
    """
    return get_model(model_name="da_lm_3gram_folketinget.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="011771d8bef6ff531812a768f631b4a2",
                     cache_dir=cache_dir,
                     file_type="language_model")
