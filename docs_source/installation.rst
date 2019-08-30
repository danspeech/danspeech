============
Installation
============

DanSpeech is installed by

.. code-block:: bash

    $ pip install danspeech


If you wish to install from source, then do

.. code-block:: bash

    $ git clone https://github.com/danspeech/danspeech
    $ cd danspeech
    $ pip install .


DanSpeech requires python 3.5+.

CTC-decode
----------
To use language models with the system, you will need to additionally install `ctc-decode <https://github.com/parlance/ctcdecode.git>`_.

.. code-block:: bash

    $ git clone --recursive https://github.com/parlance/ctcdecode.git
    $ cd ctcdecode && pip install .

**Warning:** This might prove rather troublesome to install on a windows system.

.. _pyaudio-install:

PyAudio
-------
If you wish to transcribe a stream of audio e.g. from a microphone, you will need to additionally install
`PyAudio <https://pypi.org/project/PyAudio/>`_.