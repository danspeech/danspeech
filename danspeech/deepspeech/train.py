import os
import sys

import time
import argparse
import random
import numpy as np

import torch
import torch.distributed as dist

from danspeech.audio.datasets import BatchDataLoader, DanSpeechDataset
from danspeech.audio.parsers import SpectrogramAudioParser
from torch.utils.data.distributed import DistributedSampler

from danspeech.audio.augmentation import DanSpeechAugmenter
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.deepspeech.model import DeepSpeech
from danspeech.deepspeech.utils import TensorBoardLogger, AverageMeter, reduce_tensor, sum_tensor
from danspeech.errors.training_errors import ArgumentMissingForOption

import distiller
import distiller.apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger


def train_model(model, args, package=None):
    """
    :param model: a DeepSpeech model, can either be a predefined DanSpeech model or a new model build by the user
    :param args: ToDO: go nuts optional and required arguments, we are removing the argparsing
    :param package: Serialized filed to load a model from
    :return: Accuracy scores and a serialized model object, saved as a .pth file
    """
    print(args)
    main_proc = True

    # Handle all arguments
    device = torch.device("cuda" if args.cuda else "cpu")

    # Model save folder (make sure dir exists)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.log_training and main_proc:
        # Log dir (make sure dir exists)
        log_dir = args.visualize_folder + args.id
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_logger = TensorBoardLogger(args.id, log_dir, args.log_params)

    # Model path
    model_path = args.save_folder + args.id + ".pth"

    # labels
    labels = model.labels

    # Audio configuration
    audio_conf = model.audio_conf

    # Distributed training
    if args.distributed:
        # toDO: Add try catch and give proper error message
        from apex.parallel import DistributedDataParallel

        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        main_proc = args.rank == 0  # main_process is handling saving and logging

    # Init Tensors to track the results
    # ToDO: Consider making them numpy? No reason for them being tensors
    loss_results = torch.Tensor(args.epochs)
    cer_results = torch.Tensor(args.epochs)
    wer_results = torch.Tensor(args.epochs)

    best_wer = None

    avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    # If continuing training, then we simply continue training the data of the saved model
    if args.continue_training:
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

            if main_proc and args.log_training:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values(start_epoch, package)

    print(model)
    decoder = GreedyDecoder(labels)
    model = model.to(device)

    # Init audio parsers and datasets
    augmenter = None
    if args.with_augmentations:
        augmenter = DanSpeechAugmenter(sampling_rate=audio_conf["sampling_rate"])

    train_audio_parser = SpectrogramAudioParser(audio_config=audio_conf, data_augmenter=augmenter)

    train_dataset = DanSpeechDataset(args.train_data_path, labels=labels, audio_parser=train_audio_parser)

    validation_audio_parser = SpectrogramAudioParser(audio_config=audio_conf)
    validation_dataset = DanSpeechDataset(args.validation_data_path, labels=labels,
                                          audio_parser=validation_audio_parser)

    # Initialize batch loaders
    if args.distributed:

        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)

        train_batch_loader = BatchDataLoader(train_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=train_sampler,
                                             pin_memory=True)

        validation_sampler = DistributedSampler(validation_dataset, num_replicas=args.world_size, rank=args.rank)

        validation_batch_loader = BatchDataLoader(validation_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  sampler=validation_sampler)

        model = DistributedDataParallel(model)

    else:
        train_batch_loader = BatchDataLoader(train_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=True, pin_memory=True)

        validation_batch_loader = BatchDataLoader(validation_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False)

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    print(model)



    ####################
    # Above this line is fix

    try:
        import CT
    except ModuleNotFoundError as e:
        from torch.nn.modules.loss import CTCLoss

    criterion = CTCLoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and epoch != 0:
            train_sampler.set_epoch(epoch)

        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_batch_loader, start=start_iter):

            if i == len(train_batch_loader):
                break
            inputs, targets, input_percentages, target_sizes, corr_status = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            float_out = out.float()
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            inf = float("inf")
            if args.distributed:
                loss_value = reduce_tensor(loss, args.world_size).item()
            else:
                loss_value = loss.item()
            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (args.epochs), (i + 1), len(train_batch_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                wer_results_corr=wer_results_corr, cer_results_corr=cer_results_corr,
                                                wer_results_ncorr=wer_results_ncorr,
                                                cer_results_ncorr=cer_results_ncorr,
                                                avg_loss=avg_loss,
                                                distributed=args.distributed),
                           file_path)

            del loss, out, float_out

        avg_loss /= len(train_batch_loader)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer_corr, total_cer_ncorr, total_wer_corr, total_wer_ncorr = 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(validation_batch_loader), total=len(validation_batch_loader)):
                inputs, targets, input_percentages, target_sizes, corr_status = data  # -- added stuff here
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                # unflatten targets
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

                # -- added stuff here
                wer_corr, wer_ncorr, cer_corr, cer_ncorr = 0, 0, 0, 0
                for x in range(len(target_strings)):
                    if corr_status[x][0] == 0:
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer_ncorr += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer_ncorr += decoder.cer(transcript, reference) / float(len(reference))
                    else:
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer_corr += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer_corr += decoder.cer(transcript, reference) / float(len(reference))

                total_cer_corr += cer_corr
                total_cer_ncorr += cer_ncorr
                total_wer_corr += wer_corr
                total_wer_ncorr += wer_ncorr
                del out

            # -- added stuff
            if args.distributed:
                # Sums tensor across all devices
                total_wer_corr_tensor = torch.tensor(total_wer_corr).to(device)
                total_wer_corr_tensor = sum_tensor(total_wer_corr_tensor)
                total_wer_corr = total_wer_corr_tensor.item()

                total_wer_ncorr_tensor = torch.tensor(total_wer_ncorr).to(device)
                total_wer_ncorr_tensor = sum_tensor(total_wer_ncorr_tensor)
                total_wer_ncorr = total_wer_ncorr_tensor.item()

                total_cer_corr_tensor = torch.tensor(total_cer_corr).to(device)
                total_cer_corr_tensor = sum_tensor(total_cer_corr_tensor)
                total_cer_corr = total_cer_corr_tensor.item()

                total_cer_ncorr_tensor = torch.tensor(total_cer_ncorr).to(device)
                total_cer_ncorr_tensor = sum_tensor(total_cer_ncorr_tensor)
                total_cer_ncorr = total_cer_ncorr_tensor.item()

                del total_cer_corr_tensor, total_cer_ncorr_tensor, total_wer_corr_tensor, total_wer_ncorr_tensor

            wer_corr = total_wer_corr / len(test_batch_loader.dataset)
            wer_ncorr = total_wer_ncorr / len(test_batch_loader.dataset)
            cer_corr = total_cer_corr / len(test_batch_loader.dataset)
            cer_ncorr = total_cer_ncorr / len(test_batch_loader.dataset)

            wer_corr *= 100
            wer_ncorr *= 100
            cer_corr *= 100
            cer_ncorr *= 100

            loss_results[epoch] = avg_loss
            wer_results_corr[epoch] = wer_corr
            wer_results_ncorr[epoch] = wer_ncorr
            cer_results_corr[epoch] = cer_corr
            cer_results_ncorr[epoch] = cer_ncorr

            print('Validation Summary Epoch: [{0}]\t'
                  'Average correlated WER {werc:.3f}\t'
                  'Average not-correlated WER {wernc:.3f}\t'
                  'Average correlated CER {cerc:.3f}\t'
                  'Average not-correlated CER {cernc:.3f}\t'.format(epoch + 1, werc=wer_corr, wernc=wer_ncorr,
                                                                    cerc=cer_corr, cernc=cer_ncorr))

            if args.tensorboard and main_proc:
                values = {
                    'loss_results': loss_results,
                    'cer_results_corr': cer_results_corr,
                    'cer_results_ncorr': cer_results_ncorr,
                    'wer_results_corr': wer_results_corr,
                    'wer_results_ncorr': wer_results_ncorr
                }
                if args.tensorboard and main_proc:
                    tensorboard_logger.update(epoch, values, model.named_parameters())

            if args.checkpoint and main_proc:
                file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                wer_results_corr=wer_results_corr, cer_results_corr=cer_results_corr,
                                                wer_results_ncorr=wer_results_ncorr,
                                                cer_results_ncorr=cer_results_ncorr,
                                                distributed=args.distributed),
                           file_path)
                # anneal lr
                param_groups = optimizer.param_groups
                for g in param_groups:
                    g['lr'] = g['lr'] / args.learning_anneal
                print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            if main_proc and (best_wer is None or best_wer > wer_ncorr):
                print("Found better validated model, saving to %s" % model_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                wer_results_corr=wer_results_corr, cer_results_corr=cer_results_corr,
                                                wer_results_ncorr=wer_results_ncorr,
                                                cer_results_ncorr=cer_results_ncorr,
                                                distributed=args.distributed)
                           , model_path)

                best_wer = wer_ncorr
                avg_loss = 0
