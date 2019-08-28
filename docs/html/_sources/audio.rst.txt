=============
Audio Sources
=============

Using the correct input for the speech recognition models is very important. If you decide to use your own method
of loading data (or ``librosa``'s method), then recognition performance might/will be degraded.

A list of the most important audio utilities is provided below.

.. automodule:: danspeech.audio
    :members: load_audio, load_audio_wavPCM, Microphone
    :exclude-members: MicrophoneStream
