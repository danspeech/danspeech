import torch
import librosa
from danspeech.deepspeech.model import DeepSpeech
from danspeech.audio.parsers import SpectrogramAudioParser

from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy


package = torch.load('/home/mcn/newModels/test_lstm_quantization.pth', map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)
model.convert_lstm()

print(model)

quantizer = PostTrainLinearQuantizer(
    deepcopy(model),
    model_activation_stats='./lstm_pretrained_stats.yaml'
)

stats_before_prepare = deepcopy(quantizer.model_activation_stats)
y, sr = librosa.load("/home/mcn/danspeech/example_files/u0013002.wav", sr=16000)
parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
dummy_input = parser.parse_audio(y)
print(dummy_input.size())
max_length = dummy_input.size(1)
dummy_input = (torch.zeros(161, 419, 161, 419), torch.tensor(419))
print(dummy_input)
quantizer.prepare_model(dummy_input)

print(quantizer.model)