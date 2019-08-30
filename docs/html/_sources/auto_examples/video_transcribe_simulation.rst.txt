.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_video_transcribe_simulation.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_video_transcribe_simulation.py:


Simulates transcribing a video with DanSpeech
=============================================

This is an example where a full video (converted to .wav) is being transcribed as it would be transcribed
in a setting where the input was a source with chunk size 1024 (think of a microphone).

This specific example was used to transcribe a
`"Udvalgsmøde" from Folketinget (Danish Parliament) <https://www.ft.dk/aktuelt/webtv/video/20182/beu/td.1583453.aspx?as=1>`_
with offset 18 seconds.

The result can be seen in `Udvalgsmøde result <https://gist.github.com/Rasmusafj/fb416032f70331a5641446bb0e61d008>`_

Note that the video needs to be converted to the correct format. See below:

- First convert your video to .wav in the correct format
- Example using ffmpeg:

.. code-block:: bash

    ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 video_audio.wav


.. code-block:: default



    from danspeech import Recognizer
    from danspeech.pretrained_models import Folketinget
    from danspeech.language_models import Folketinget3gram
    from danspeech.audio import load_audio

    import numpy as np

    import argparse


    # First convert your video to .wav in the correct format
    # Example using ffmpeg:
    # ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 video_audio.wav
    parser = argparse.ArgumentParser(description='DanSpeech simulate transcribing video')
    parser.add_argument('--wav-path', type=str, required=True,  help='Path to the .wav file which you wish to transcribe')
    parser.add_argument('--gpu', action='store_true', help='Whether to use danspeech with GPU (Recommended)')
    parser.add_argument('--offset', type=int, default=0, help='Offset in seconds. Start video not from start')
    parser.add_argument('--outfile', type=str, default="", required=False,
                        help='path to write the results. (from where the script is run)')


    def pretty_format_seconds(seconds):
        minutes = str(int(seconds / 60))
        remaining_seconds = str(int(seconds % 60))
        if int(remaining_seconds) < 10:
            remaining_seconds = "0" + remaining_seconds

        return "{0}:{1}".format(minutes, remaining_seconds)


    if __name__ == '__main__':
        args = parser.parse_args()
        model = Folketinget()
        lm = Folketinget3gram()
        recognizer = Recognizer(model=model, lm=lm, with_gpu=args.gpu, alpha=1.0471119809697471,
                                beta=2.8309374387487924, beam_width=64)

        # Load audio and do an offset start
        audio = load_audio(args.wav_path)
        offset = args.offset * 16000
        audio = audio[offset:]

        # Energy threshold. This can vary a lot depending on input
        energy_threshold = 600

        # Simulate a stream with chunk size 1024
        step = 1024
        iterator = 0

        # seconds of non-speaking audio before a phrase is considered complete
        pause_threshold = 0.55

        pause_buffer_count = np.ceil(pause_threshold / (1024 / 16000))

        # seconds of speaking before considering it a phrase
        phrase_threshold = 0.2
        phrase_buffer_count = np.ceil(phrase_threshold / (1024 / 16000))

        # Control variables
        is_speaking = False
        frames_counter = 0
        pause_count = 0

        if args.outfile:
            f = open(args.outfile, "w", encoding="utf-8")

        # Main loop
        while (iterator + step) < len(audio):

            # Get data
            temp_data = audio[iterator:iterator + step]

            # Simple energy measure
            energy = np.sqrt((temp_data * temp_data).sum() / (1. * len(temp_data)))

            # If energy is above, then speaking has started
            if energy > energy_threshold and not is_speaking:
                # General requirements for start
                is_speaking = True

                # We give the previous ~0.120 seconds with the output i.e. two frames just in case.
                start_index = iterator - 2*step

                # Must not be negative though
                if start_index < 0:
                    start_index = iterator

            # add to iterator here!
            iterator += step

            if is_speaking:
                frames_counter += 1

                # Control whether we should stop
                if energy > energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1

            # This indicaes we should stop!
            if pause_count > pause_buffer_count and is_speaking:  # end of the phrase

                # now check how long the spoken utterance was disregarding the "not enough energy" pause count
                if (frames_counter - pause_count) > phrase_buffer_count:
                    trans = recognizer.recognize(audio[start_index:iterator])

                    # samples --> seconds --> pretty format
                    start = pretty_format_seconds((start_index + offset) / 16000)
                    end = pretty_format_seconds((iterator + offset) / 16000)
                    out_string = "start: {0}, end: {1}, transcription: {2}".format(start, end, trans)
                    print(out_string)

                    if args.outfile:
                        f.write(out_string + "\n")

                is_speaking = False
                frames_counter = 0
                pause_count = 0

        f.close()

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_auto_examples_video_transcribe_simulation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: video_transcribe_simulation.py <video_transcribe_simulation.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: video_transcribe_simulation.ipynb <video_transcribe_simulation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
