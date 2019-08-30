.. _training-repo:

=============================
DanSpeech training repository
=============================
The DanSpeech project provides a repository for training new DanSpeech models which resides here: `danspeech_training <https://github.com/danspeech/danspeech_training>`_.
The training repo does not have extensive documentation, but does include a readme file with the most important information. Additionally the code is commented and should be considered the main source of documentation.

We consider the training repository to be an extension of the package, primarily targeting developers who want to experiment with training or finetuning DanSpeech models.
The repository provides a fairly intuitive training script with three train wrappers for training new models, finetuning models and continue a training pass.
In its current state the training script relies on a set of pre-defined parameters, but in a future release we will extend the repo with a branch relying on argparse instead, as this significantly
simplifies multi-GPU training. The repository furthermore provides a script for testing trained models on a held-out test-set.

We assume a specific structure of the data folder for training and testing, for more information see the `readme <https://github.com/danspeech/danspeech_training/blob/master/README.md>`_.

The training repo relies on functionality from DanSpeech, and as such having DanSpeech installed is mandatory for using danspeech_training.

