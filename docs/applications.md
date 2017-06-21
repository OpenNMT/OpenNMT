## Machine Translation

Neural Machine Translation (NMT) is the default and original task for OpenNMT. It requires a corpus of bilingual sentences (for instance as available on [Opus](http://opus.lingfil.uu.se) for a very large variety of domains and language pairs). Training a NMT engine is a 3 steps process:

* [Tokenization](/tools/tokenization/)
* Preprocessing
* Training

Step by step process is described on the [quickstart page](/quickstart/) and full process to train large system is described on the forum [here](http://forum.opennmt.net/t/training-english-german-wmt15-nmt-engine/29).

![Neural Machine Translation](../img/nmt.png)

## Summarization

Summarization models are trained exactly like NMT models - the nature of the training data is different: source corpus are full length document / articles, and target are summaries. [This forum post](http://forum.opennmt.net/t/text-summarization-on-gigaword-and-rouge-scoring/85/) details how to train and evaluation a summarization model.

## Im2Text

Im2Text, written by Yuntian Deng from HarvardNLP group, is implementing a generic image-to-text application on top of OpenNMT libraries for [visual markup decompilation](https://arxiv.org/pdf/1609.04938v1.pdf). Main modification to generic OpenNMT is the encoder introducing CNN layers in combination with RNN.

## Speech Recognition

While OpenNMT is not primarily targetting speech recognition applications, its ability to support [input vectors](/data/preparation/#input-vectors) and [pyramidal RNN](/training/models/#pyramidal-deep-bidirectional-encoder) make possible end-to-end experiments on speech to text applications as described for instance in [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211). See for instance [monophone speech recognition with OpenNMT](http://forum.opennmt.net/t/monophone-speech-recognition-with-opennmt/542) on the forum.

![Listen, Attend and Spell](../img/las.png)

## Sequence Tagging

A Sequence Tagger is available in OpenNMT. Sharing the same encoder code, it does not need a decoder, since each input is synced with an output - a sequence tagger just needs an encoder and a generation layer. Sequence Tagging can be used for any annotation task such as part of speech tagging.

![Sequence Tagger](../img/seqtagger.png)

To train a sequence tagger:

* prepare the bitext with source and target sequence length identical (you can use `-check_plength` option in preprocessor).
* train the model with `-model_type seqtagger`
* use the model with `tag.lua`

## Language Modelling

A language model is very similar to a sequence tagger - the main difference is that the output "tag" for each token is the following word in source sentence.

![Language Model](../img/lm.png)

* prepare data with `-data_type monotext` in preprocessor
* train the model with `-model_type lm`
* use the model with `lm.lua`

