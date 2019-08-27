# DanSpeech Documentation
## What is DanSpeech?

DanSpeech is an open-source Danish speech recognition (speech-to-text) python package based on the 
[PyTorch](https://pytorch.org/) deep learning framework. It was developed as part of a Master's thesis at DTU 
by Martin Carsten Nielsen and Rasmus Arpe Figh Jensen with supervisor Professor Lars Kai. 

All of the DanSpeech models are end-to-end [DeepSpeech 2](https://arxiv.org/abs/1512.02595) models trained on danish
text with various data agumentations as an attempt multiply the rather small and public speech recognition 
data available in Danish. The models perform state-of-the-art speech recognition in Danish but performance is
 evidently not perfect and conditioned on specific use-cases.

Danspeech contains:
* An easy-to-use Recognizer that supports different use-cases for Danish speech recognition.
* Pre-trained models.
* Pre-trained language models.


## Motivation
We believe that speech recognition in Danish should be available for anyone. We therefore developed
an open-source and easy-to-use speech recognition system for Danish. 

An open-source speech recognition system is important to ensure that Danish speech recognition performance 
will not continue to fall miles behind the performance of English systems. Speech recognition will inevitable
be a big part of future IT innovations. Without an easy-to-use and free system, innovation in Danish technologies 
utilization speech recognition is hindered.


The system can be used freely commercially used for companies without the resources to develop their own speech recognition 
systems or companies who do not wish to outsource speech recognition.

The system can also be used as part of education at various universities in Denmark.


## Performance
We benchmarked the system on two Danish benchmarks, namely the public available [Nordisk Språkteknologi](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en)
(NST) dataset and our own (not public due to data gathering restrictions and GDPR) DanSpeech dataset (~1000 noisy recordings).
The performance is evaulated in [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) (WER)

| Dataset           | Models                                       | Performance |
|-------------------|---------------------------------------------|-------------|
| NST test          | DanSpeechPrimary + DSL5Gram (not pruned) LM | 12.85% WER  |
| DanSpeech dataset | TransferLearned + DSL5Gram (not pruned) LM  | 25.75% WER  |



If you know the specific domain which you wish to apply Danish speech recognition to, and you have either text resources 
or in the best case speech data available, then you can finetune DanSpeech models to achieve a much better performance for your
specific domain than the performance DanSpeech pre-trained models are capable of.

See [DanSpeech training repository](#danspeech-training-repository) and [DanSpeech Demo](#danspeech-demo). 

## DanSpeech training repository
If you wish to train your own speech recognition models, or finetune DanSpeech models to your specific
domain/use-case, then use the DanSpeech training repository. This requires technical expertise and a GPU. 

As mentioned above, you can finetune DanSpeech models to achieve a much better performance for your
specific domain than the performance DanSpeech pre-trained models are capable of.



## DanSpeech Demo
To test the DanSpeech models on your own audio files or on your own speech, we additionally created a demo 
that runs as a development django server on localhost. It is easy to install and hence easy to test the models
with a GUI (little technical knowledge is required to play around with the demo).

The demo also features the finetuned Folketinget DanSpeech model with the finetuned Folketinget language model
transcribing a Danish Parliament meeting, which the model was not trained on.

