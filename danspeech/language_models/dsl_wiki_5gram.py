from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def DSLWiki5gram(cache_dir=None):
    """
    DSL and wikipedia corpus trained 5-gram model.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="dsl_wiki_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="070287617eacbbde79df2be34ac9615f",
                     cache_dir=cache_dir,
                     file_type="language_model")
