from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def Folketinget3gram(cache_dir=None):
    """
    3-gram language model trained on all meeting summaries from the Danish Parliament (Folketinget)

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="da_lm_3gram_folketinget.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="011771d8bef6ff531812a768f631b4a2",
                     cache_dir=cache_dir,
                     file_type="language_model")
