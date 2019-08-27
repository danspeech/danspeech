from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def DSLWiki3gram(cache_dir=None):
    """
    DSL and wikipedia corpus trained 3-gram model.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="dsl_wiki_3gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="f38f55a1e14ad888cee3ea1e643593dc",
                     cache_dir=cache_dir,
                     file_type="language_model")
