import torch
from danspeech.deepspeech.model import DeepSpeech
from danspeech.audio.parsers import SpectrogramAudioParser
from danspeech.audio.datasets import BatchDataLoader, DanSpeechDataset

from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy


package = torch.load('/home/s123106/danish-speech-recognition/models/test_lstm_quantization.pth', map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)
model.convert_lstm()

print(model)

quantizer = PostTrainLinearQuantizer(
    deepcopy(model),
    model_activation_stats='./lstm_pretrained_stats.yaml'
)

stats_before_prepare = deepcopy(quantizer.model_activation_stats)
validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
validation_set = DanSpeechDataset("/scratch/s134843/preprocessed_validation/", labels=model.labels, audio_parser=validation_parser)
validation_batch_loader = BatchDataLoader(validation_set, batch_size=1, shuffle=False)

for i, (data) in enumerate(validation_batch_loader):
    if i == 1:
        break
    inputs, targets, input_percentages, target_sizes = data
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

    dummy_input = (inputs, input_sizes)

quantizer.prepare_model(dummy_input)

print(quantizer.model)