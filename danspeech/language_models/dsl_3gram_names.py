from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "https://github.com/Rasmusafj/models_development/raw/master/dsl_names.klm"


def DSL3gramWithNames(cache_dir=None):
    """
    Includes DSL + a bias towards the most common names in Denmark.

    DSL 3-gram language model. This is the best performing for out test cases along with DSL 5-gram.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="dsl_names.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="1b47e2db841c6be5c62004ef51a40c68",
                     cache_dir=cache_dir,
                     file_type="language_model")
