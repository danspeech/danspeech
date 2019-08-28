"""
Stream of audio from your microphone
====================================

This is an example of using your own Microphone to continuously transcribe what is being uttered. Whenever the
recognizer detects a silence in the audio stream from your microphone, it will be transcribed.
"""
from danspeech import Recognizer
from danspeech.pretrained_models import TransferLearned
from danspeech.audio.resources import Microphone
from danspeech.language_models import DSL3gram


# Get a list of microphones found by PyAudio
mic_list = Microphone.list_microphone_names()
mic_list_with_numbers = list(zip(range(len(mic_list)), mic_list))
print("Available microphones: {0}".format(mic_list_with_numbers))

# Choose the microphone
mic_number = input("Pick the number of the microphone you would like to use: ")

# Init a microphone object
m = Microphone(sampling_rate=16000, device_index=int(mic_number))

# Init a DanSpeech model and create a Recognizer instance
model = TransferLearned()
recognizer = Recognizer(model=model)

# Try using the DSL 3 gram language model
try:
    lm = DSL3gram()
    recognizer.update_decoder(lm=lm)
except ImportError:
    print("ctcdecode not installed. Using greedy decoding.")

print("Speek a lot to adjust silence detection from microphone...")
with m as source:
    recognizer.adjust_for_speech(source, duration=5)

# Enable streaming
recognizer.enable_streaming()

# Create the streaming generator which runs a background thread listening to the microphone stream
generator = recognizer.streaming(source=m)

# The below code runs for a long time. The generator returns transcriptions of spoken speech from your microphone.
print("Speak")
for i in range(100000):
    trans = next(generator)
    print(trans)
