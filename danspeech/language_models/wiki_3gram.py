from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def Wiki3gram(cache_dir=None):
    """
    wikipedia corpus trained 3-gram model.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="wiki_3gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="12877123bbbbaa72826746cad0af6f7d",
                     cache_dir=cache_dir,
                     file_type="language_model")
