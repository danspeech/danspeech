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

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
sys.argv.pop(1)

print(os.environ["CUDA_VISIBLE_DEVICES"])

if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

# -- DanSpeech-related arguments
parser = argparse.ArgumentParser(description='DanSpeech Training / Pruning')

# -- Data-specific arguments, required!
parser.add_argument('--train-data-path', metavar='DIR',
                    help='Path to training data', default=None)
parser.add_argument('--validation-data-path', metavar='DIR',
                    help='Path to validation data', default=None)
parser.add_argument('--labels-path', default='./labels.json',
                    help='Contains all characters for transcription')

# -- Spectogram hyper-parameters, defaults to DanSpeech model parameters
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

# -- Network-specific hyper-parameters, defaults to DanSpeech baseline model parameters
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=1.0, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')

# -- Model-monitoring related arguments
parser.add_argument('--id', default='DanSpeech_dummy_model',
                    help='File name given to checkpoint models')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing',
                    default=True)

# -- GPU-training arguments
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--num-workers', default=6, type=int, help='Number of workers used in data-loading')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=1337, type=int, help='Seed to generators')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')

# -- Training specific arguments
parser.add_argument('--continue-from', default='', help='Continue training from checkpoint')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint by freezing all layers except the output layer')

# -- Multiple-GPU training arguments
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

# -- Distiller-related pruning arguments
parser.add_argument('--prune', dest='prune', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (dafault is to use hard-coded schedule)')


def train_model(model, args, package=None):
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
        augmenter = DanSpeechAugmenter(sampling_rate=audio_conf["sample_rate"])

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
        import torch.nn.modules.loss.CTCLoss as CTCLoss

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


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    args.cuda = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Begin training
    print(args)
    train_model(args)
