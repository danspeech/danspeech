from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "toDo"


def Folketinget3gram(cache_dir=None):
    return get_model(model_name="da_lm_3gram_folketinget.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="b1472816d6217a0ee06cbc540d4c63df",
                     cache_dir=cache_dir,
                     file_type="language_model")
