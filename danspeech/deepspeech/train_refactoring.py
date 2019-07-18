import os
import time
import tqdm
import warnings

import torch
from torch.nn.modules.loss import CTCLoss

from danspeech.audio.datasets import BatchDataLoader, DanSpeechDataset
from danspeech.audio.parsers import SpectrogramAudioParser
from danspeech.audio.augmentation import DanSpeechAugmenter
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.deepspeech.utils import TensorBoardLogger, AverageMeter, reduce_tensor, sum_tensor
from danspeech.errors.training_errors import ArgumentMissingForOption


class NoModelSaveDirSpecified(Warning):
    pass


class NoLoggingDirSpecified(Warning):
    pass


def _train_model(model, train_data_path, validation_data_path, model_id, model_save_dir=None, tensorboard_log_dir=None,
                 continue_train=False, augmented_training=True, distributed=False, batch_size=32, num_workers=6,
                 cuda=False, lr=3e-4, momentum=0.9, weight_decay=1e-5, epochs=20, compression_scheduler=None,
                 max_norm=400, package=None):
    if not model_save_dir:
        warnings.warn("You did not specify a directory for saving the trained model."
                      "Defaulting to ~/.danspeech/custom/ directory.", NoModelSaveDirSpecified)

        model_save_dir = os.path.join(os.path.expanduser('~'), '.danspeech', "custom")

    os.makedirs(model_save_dir, exist_ok=True)

    if tensorboard_log_dir:
        logging_process = True
        tensorboard_logger = TensorBoardLogger(model_id, tensorboard_log_dir)
    else:
        logging_process = False
        warnings.warn(
            "You did not specify a directory for logging training process. Training process will not be logged.",
            NoLoggingDirSpecified)

    loss_results = torch.Tensor(epochs)
    cer_results = torch.Tensor(epochs)
    wer_results = torch.Tensor(epochs)

    # -- prepare model for processing
    device = torch.device("cuda" if cuda else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                nesterov=True, weight_decay=weight_decay)

    # -- initialize metrics
    avg_loss = 0
    start_epoch = 0
    start_iter = 0

    if continue_train:
        if not package:
            raise ArgumentMissingForOption("If you want to continue training, please support a package with previous"
                                           "training information or use the finetune option instead")
        else:
            optim_state = package['optim_dict']
            optimizer.load_state_dict(optim_state)
            start_epoch = int(package['epoch']) - 1  # Index start at 0 for training

            print("Previous Epoch: {0}".format(start_epoch))

            start_epoch += 1
            start_iter = 0

            avg_loss = int(package.get('avg_loss', 0))
            loss_results_ = package['loss_results']
            cer_results_ = package['cer_results']
            wer_results_ = package['wer_results']

            # ToDo: Make depend on the epoch from the package
            previous_epochs = loss_results_.size()[0]
            print("Previously ran: {0} epochs".format(previous_epochs))

            loss_results[0:previous_epochs] = loss_results_
            wer_results[0:previous_epochs] = cer_results_
            cer_results[0:previous_epochs] = wer_results_

            if logging_process:
                tensorboard_logger.load_previous_values(start_epoch, package)

    # -- initialize audio parser and dataset
    if augmented_training:
        augmenter = DanSpeechAugmenter(sampling_rate=model.audio_conf["sampling_rate"])
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

    decoder = GreedyDecoder(model.labels)
    criterion = CTCLoss()
    model = model.to(device)

    # -- verbatim training outputs during progress
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    print("Initializations complete, starting training pass on model:\n")
    print(model)
    try:
        for epoch in range(start_epoch, epochs):
            print('started training epoch %s', epoch + 1)
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
            loss_results[epoch] = avg_loss
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

            if logging_process:
                logging_values = {
                    "loss_results": loss_results,
                    "wer": avg_wer_epoch,
                    "cer": avg_cer_epoch
                }
                tensorboard_logger.update(epoch, logging_values)

            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=avg_wer_epoch, cer=avg_cer_epoch))

            # -- reset start iteration for next epoch
            start_iter = 0

    except KeyboardInterrupt:
        # ToDO: added distributed processing
        print('Implement a DanSpeech exception for early stopping here')


def train_new(model, train_data_path, validation_data_path, model_id, model_save_dir=None,
              tensorboard_log_dir=None, **args):

    _train_model(model, train_data_path, validation_data_path, model_id, model_save_dir=model_save_dir,
                 tensorboard_log_dir=tensorboard_log_dir, **args)


def finetune(model, train_data_path, validaton_data_path, model_id, model_save_dir=None,
             tensorboard_log_dir=None, **args):

    _train_model(model, train_data_path, validaton_data_path, model_id, model_save_dir=model_save_dir,
                 tensorboard_log_dir=tensorboard_log_dir, **args)


def continue_training(model, train_data_path, validaton_data_path, model_id, package, model_save_dir=None,
                      tensorboard_log_dir=None, **args):

    _train_model(model, train_data_path, validaton_data_path, model_id, model_save_dir=model_save_dir,
                 tensorboard_log_dir=tensorboard_log_dir, continue_train=True, package=package, **args)
