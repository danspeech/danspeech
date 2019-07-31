import danspeech
import time
from danspeech.audio.resources import SpeechFile

import torch
from danspeech.deepspeech.model import DeepSpeech

import distiller
from distiller import sparsity


# -- pruned baseline model
package = torch.load('/home/mcn/newModels/baidu_pruned_baseline.pth', map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)


num_layers = 0
total_sparsity = 0
for name, layer in model.named_parameters():
    if sparsity(layer) > 0:
        total_sparsity += sparsity(layer)
        num_layers += 1
        layer = layer.to_sparse()

total_sparsity = total_sparsity / num_layers

recognizer = danspeech.Recognizer(model=model)

with SpeechFile(filepath="../example_files/u0042019.wav") as source:
    audio = recognizer.record(source)

start = time.time()
output = recognizer.recognize(audio)
end = time.time()

print(output, "\n", "Pruned model processing time: %s" %(end-start), "\n", "Total sparsity: %s" %total_sparsity, "\n")


# -- baseline model
package_baseline = torch.load('/home/mcn/newModels/DanSpeech_4gpu_128batch_200epochs.pth', map_location=lambda storage, loc: storage)
model_baseline = DeepSpeech.load_model_package(package_baseline)

num_layers = 0
total_sparsity = 0
for name, layer in model_baseline.named_parameters():
    if sparsity(layer) > 0:
        total_sparsity += sparsity(layer)
        num_layers += 1

if num_layers != 0:
    total_sparsity = total_sparsity / num_layers
else:
    total_sparsity = 0

recognizer_baseline = danspeech.Recognizer(model=model_baseline)

with SpeechFile(filepath="../example_files/u0042019.wav") as source:
    audio = recognizer_baseline.record(source)

start = time.time()
output_baseline = recognizer_baseline.recognize(audio)
end = time.time()

print(output_baseline, "\n", "Baseline model processing time: %s" %(end-start), "\n", "Total sparsity: %s" %total_sparsity, "\n")


# -- distiller LSTM model
package_lstm = torch.load('/home/mcn/newModels/test_lstm_quantization.pth', map_location=lambda storage, loc: storage)
model_lstm = DeepSpeech.load_model_package(package_lstm)
model_lstm.convert_lstm()

num_layers = 0
total_sparsity = 0
for name, layer in model_lstm.named_parameters():
    if sparsity(layer) > 0:
        total_sparsity += sparsity(layer)
        num_layers += 1

if num_layers != 0:
    total_sparsity = total_sparsity / num_layers
else:
    total_sparsity = 0

recognizer_baseline = danspeech.Recognizer(model=model_lstm)

with SpeechFile(filepath="../example_files/u0042019.wav") as source:
    audio = recognizer_baseline.record(source)

start = time.time()
output_baseline = recognizer_baseline.recognize(audio)
end = time.time()

print(output_baseline, "\n", "Distiller LSTM model processing time: %s" %(end-start), "\n", "Total sparsity: %s" %total_sparsity)
