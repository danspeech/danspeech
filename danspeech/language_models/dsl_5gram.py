from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def DSL5gram(cache_dir=None):
    """
    DSL 5-gram language model. This is the best performing for out test cases along with DSL 3-gram.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="dsl_5gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="f2929d6d154b57b8be0c05347036c7e6",
                     cache_dir=cache_dir,
                     file_type="language_model")
