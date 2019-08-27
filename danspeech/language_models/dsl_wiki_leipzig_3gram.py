from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "ToDO"


def DSLWikiLeipzig3gram(cache_dir=None):
    """
    DSL, wikipedia and Leipzig corpus trained 3-gram model.

    :param cache_dir: If you wish to cash your models somewhere else than default
    :return: string, path to .klm language model
    """
    return get_model(model_name="dsl_wiki_leipzig_3gram.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="8409a469be718209afdd18692a2d5609",
                     cache_dir=cache_dir,
                     file_type="language_model")
