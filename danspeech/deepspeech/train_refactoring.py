import time
import tqdm

import torch
from torch.nn.modules.loss import CTCLoss

from danspeech.audio.datasets import BatchDataLoader, DanSpeechDataset
from danspeech.audio.parsers import SpectrogramAudioParser
from danspeech.audio.augmentation import DanSpeechAugmenter
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.deepspeech.utils import TensorBoardLogger, AverageMeter, reduce_tensor, sum_tensor


def train_model(model, train_data_path, validation_data_path, training_scheme='fine_tune', augmented_training=True,
                distributed=False, batch_size=32, num_workers=6, cuda=True, lr=3e-4, momentum=0.9, weight_decay=1e-5,
                epochs=20, compression_scheduler=None, max_norm=400):
    # -- toDO: include model diagnostic check to ensure all required parameters are included, and raise a warning
    # --       otherwise

    # -- initialize audio parser and dataset
    if augmented_training:
        augmenter = DanSpeechAugmenter(sampling_rate=model.audio_conf["sample_rate"])
    else:
        augmenter = None

    # -- audio parsers
    training_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=augmenter)
    validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)

    # -- instantiate data-sets
    training_set = DanSpeechDataset(train_data_path, labels=model.labels, audio_parser=training_parser)
    validation_set = DanSpeechDataset(validation_data_path, labels=model.labels, audio_parser=validation_parser)

    # -- initialize batch loaders
    if not distributed:
        # -- initialize batch loaders for single GPU or CPU training
        train_batch_loader = BatchDataLoader(training_set, batch_size=batch_size, num_workers=num_workers,
                                             shuffle=True, pin_memory=True)
        validation_batch_loader = BatchDataLoader(validation_set, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=False)

    # -- prepare model for processing
    device = torch.device("cuda" if cuda else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    decoder = GreedyDecoder(model.labels)
    criterion = CTCLoss()
    print("Initializations complete, starting training pass on model:\n")
    print(model)
    model = model.to(device)

    # -- initialize metrics
    loss_results, cer_results, wer_results = torch.Tensor(epochs), torch.Tensor(epochs), torch.Tensor(epochs)
    avg_loss, start_epoch, start_iter = 0, 0, 0

    # -- verbatim training outputs during progress
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    try:
        for epoch in range(0, epochs):
            print('started training epoch %s', epoch+1)
            model.train()

            # -- timings per epoch
            end = time.time()
            start_epoch_time = time.time()
            num_updates = len(train_batch_loader)

            # -- per epoch training loop, iterate over all mini-batches in the training set
            for i, (data) in enumerate(train_batch_loader, start=start_iter):
                if i == len(train_batch_loader):
                    break

                # -- grab and prepare a sample for a training pass
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # -- measure data load times, this gives an indication on the number of workers required for latency
                # -- free training.
                data_time.update(time.time() - end)

                # -- parse data and perform a training pass
                inputs = inputs.to(device)

                # -- if compression is used, activate compression schedule
                if compression_scheduler:
                    import distiller
                    import distiller.apputils as apputils
                    compression_scheduler.on_minibatch_begin(epoch, minibatch_id=i,
                                                             minibatches_per_epoch=num_updates)

                # -- compute the CTC-loss and average over mini-batch
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)
                float_out = out.float()
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                loss = loss / inputs.size(0)

                # -- check for diverging losses
                loss_value = loss.item()
                if loss_value == float("inf") or loss_value == -float("inf"):
                    print("WARNING: received an inf loss, setting loss value to 0")
                    loss_value = 0

                # -- update average loss, and loss tensor
                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                # -- if compression is used, allow the scheduler to modify the loss before the backward pass
                if compression_scheduler:
                    loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=i,
                                                                      minibatches_per_epoch=num_updates,
                                                                      loss=loss, return_loss_components=False)

                # -- compute gradients and back-propagate errors
                optimizer.zero_grad()
                loss.backward()

                # -- avoid exploding gradients by clip_grad_norm, defaults to 400
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # -- stochastic gradient descent step
                optimizer.step()

                # -- if compression is used, keep track of when to mask
                if compression_scheduler:
                    compression_scheduler.on_minibatch_end(epoch, minibatch_id=i,
                                                           minibatches_per_epoch=num_updates)

                # -- measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (epochs), (i + 1), len(train_batch_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

                del loss, out, float_out

            # -- report epoch summaries and prepare validation run
            avg_loss /= len(train_batch_loader)
            epoch_time = time.time() - start_epoch_time
            print('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

            # -- prepare validation specific parameters, and set model ready for evaluation
            total_cer, total_wer = 0, 0
            model.eval()
            with torch.no_grad():
                for i, (data) in tqdm(enumerate(validation_batch_loader), total=len(validation_batch_loader)):
                    inputs, targets, input_percentages, target_sizes = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                    # -- unflatten targets
                    split_targets = []
                    offset = 0
                    targets = targets.numpy()
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size

                    inputs = inputs.to(device)
                    out, output_sizes = model(inputs, input_sizes)
                    decoded_output, _ = decoder.decode(out, output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)

                    # -- compute accuracy metrics
                    wer, cer = 0
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))

                    total_wer += wer
                    total_cer += cer
                    del out

            # -- compute average metrics for the validation pass
            avg_wer_epoch = (total_wer / len(validation_batch_loader.dataset)) * 100
            avg_cer_epoch = (total_cer / len(validation_batch_loader.dataset)) * 100

            # -- append metrics for logging
            loss_results[epoch], wer_results[epoch], cer_results[epoch] = avg_loss, avg_wer_epoch, avg_cer_epoch

            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

            # -- reset start iteration for next epoch
            start_iter = 0

    except KeyboardInterrupt:
        # ToDO: added distributed processing
        print('Implement a DanSpeech exception for early stopping here')
