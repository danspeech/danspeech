============
Installation
============

Currently DanSpeech is installed by

.. code-block:: bash

    $ git clone https://github.com/Rasmusafj/danspeech
    $ cd danspeech
    $ pip install . -r requirements.txt


DanSpeech requires python 3.5+.

CTC-decode
----------

To use language models with the system, you will need to additionally install `ctc-decode <https://github.com/parlance/ctcdecode.git>`_.

.. code-block:: bash

    $ git clone --recursive https://github.com/parlance/ctcdecode.git
    $ cd ctcdecode && pip install .

**Warning:** This might prove rather troublesome to install on a windows system.