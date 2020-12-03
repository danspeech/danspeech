import os
import warnings
import shutil

from .Recognizer import Recognizer
from .DanSpeechRecognizer import DanSpeechRecognizer


class NoDefaultCacheDirForDanspeech(Warning):
    pass


def clean_cache():
    """
    Method used to clean danspeech cache
    """
    cache_dir = os.path.join(os.path.expanduser('~'), '.danspeech')
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    else:
        warnings.warn("The default cache dir for danspeech (~.danspeech/ did not exist. If you are"
                      "using custom cache dir, then delete it manually.", NoDefaultCacheDirForDanspeech)
