from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def Wiki5gram(cache_dir=None):
    """
    wikipedia corpus trained 5-gram model.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="wiki_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="b329e215b2fde5ffe3e2c94204f6c189",
                     cache_dir=cache_dir,
                     file_type="language_model")
