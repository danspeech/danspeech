from danspeech.deepspeech.model import DeepSpeech
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.audio.datasets import BatchDataLoader, DanSpeechDataset
from danspeech.audio.parsers import SpectrogramAudioParser

import torch
import os
import tqdm

import distiller
from distiller.data_loggers import QuantCalibrationStatsCollector, collector_context


def evaluate(model, data_source):
    decoder = GreedyDecoder(model.labels)
    model.eval()
    total_wer = 0
    for i, (data) in tqdm(enumerate(data_source), total=len(data_source)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        split_targets = []
        offset = 0
        targets = targets.numpy()
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        inputs = inputs.to("cpu")
        out, output_sizes = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = decoder.convert_to_strings(split_targets)

        wer = 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer += decoder.wer(transcript, reference) / float(len(reference.split()))

        total_wer += wer

    return total_wer


# -- load model and convert LSTM layers to distiller LSTM modules
package = torch.load('/home/mcn/newModels/test_lstm_quantization.pth', map_location=lambda storage, loc: storage)
model = DeepSpeech.load_model_package(package)
model.convert_lstm()

# -- prepare data for statistics
validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
validation_set = DanSpeechDataset("/scratch/s134843/preprocessed_validation/", labels=model.labels, audio_parser=validation_parser)
validation_batch_loader = BatchDataLoader(validation_set, batch_size=96, num_workers=num_workers, shuffle=False)

# -- attempt to grab statistics
distiller.utils.assign_layer_fq_names(model)
collector = QuantCalibrationStatsCollector(model)

if not os.path.isfile("lstm_pretrained_stats.yaml"):
    with collector_context(collector) as collector:
        val_loss = evaluate(model, validation_batch_loader)
        collector.save("lstm_pretrained_stats.yaml")



