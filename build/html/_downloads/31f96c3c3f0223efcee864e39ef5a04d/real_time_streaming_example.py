"""
Stream of audio from your microphone
====================================

This is an example of using
"""


from danspeech import Recognizer
from danspeech.pretrained_models import CPUStreamingRNN, TestModel
from danspeech.audio.resources import Microphone
from danspeech.language_models import DSL3gram

print("Loading model...")
model = CPUStreamingRNN()

mic_list = Microphone.list_microphone_names()
mic_list_with_numbers = list(zip(range(len(mic_list)), mic_list))
print("Available microphones: {0}".format(mic_list_with_numbers))
mic_number = input("Pick the number of the microphone you would like to use: ")
m = Microphone(sampling_rate=16000, device_index=int(mic_number))

r = Recognizer()

print("Adjusting energy level...")
with m as source:
    r.adjust_for_ambient_noise(source, duration=1)


seconday_model = TestModel()
r = Recognizer(model=model)
lm = DSL3gram()
r.update_decoder(lm=lm)


r.enable_real_time_streaming(streaming_model=model, string_parts=False, secondary_model=seconday_model)
generator = r.real_time_streaming(source=m)

iterating_transcript = ""
print("Speak!")
while True:
    is_last, trans = next(generator)

    # If the transcription is empty, it means that the energy level required for data
    # was passed, but nothing was predicted.
    if is_last and trans:
        print("Final: " + trans)
        iterating_transcript = ""
        continue

    if trans:
        iterating_transcript += trans
        print(iterating_transcript)
        continue



