from danspeech.utils.data_utils import get_model

LANGUAGE_MODEL_ORIGIN = "toDo"


def Folketinget3gram(cache_dir=None):
    return get_model(model_name="da_lm_3gram_folketinget.klm",
                     origin=LANGUAGE_MODEL_ORIGIN,
                     file_hash="011771d8bef6ff531812a768f631b4a2",
                     cache_dir=cache_dir,
                     file_type="language_model")
