## [Unreleased]

### Breaking changes

* Models previously trained with `-pdbrnn` or `-dbrnn` are no more compatible

### Deprecations

* `-brnn`, `-dbrnn` and `-pdbrnn` options are replaced by `-encoder_type <type>` for future extensions

### New features

* New option in preprocess to check that sizes of source and target are equal (for seqtagging)
* Add `-pdbrnn_merge` option to define how to reduce the time dimension
* Add option to segment mixed cased words
* Add options to segment words of given alphabets or when switching alphabets
* Add Google's NMT encoder

### Fixes and improvements

* SeqTagger supports input vector too
* Fix incorrect gradients when using variable length batches and bidirectional encoders

## [v0.7.1](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.7.1) (2017-05-29)

### Fixes and improvements

* Fix backward compatibility with older models using target features
* Fix importance sampling when using multiple GPUs
* Fix language models training

## [v0.7.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.7.0) (2017-05-19)

### New features

* Support vectors as inputs using [Kaldi](http://kaldi-asr.org/) input format
* Support parallel file alignment by index in addition to line-by-line
* Add script to generate pretrained word embeddings:
  * from [Polyglot](https://sites.google.com/site/rmyeid/projects/polyglot) repository
  * from pretrained *word2vec*, *GloVe* or *fastText* files
* Add an option to only fix the pretrained part of word embeddings
* Add a bridge layer between the encoder and decoder to define how encoder states are passed to the decoder
* Add `epoch_only` decay strategy to only decay learning based on epochs
* Make epoch models save frequency configurable
* Optimize decoding and training with target vocabulary reduction (importance sampling)
* [*Breaking, renamed option*] Introduce `partition` data sampling

### Fixes and improvements

* Improve command line and configuration file parser
  * space-separated list of values
  * boolean arguments
  * disallow duplicate command line options
  * clearer error messages
* Improve correctness of `DBiEncoder` and `PDBiEncoder` implementation
* Improve unicode support for languages using combining marks like Hindi
* Improve logging during preprocessing with dataset statistics
* Fix translation error of models profiled during training
* Fix translation error of models trained without attention
* Fix error when using one-layer GRU
* Fix incorrect coverage normalization formula applied during the beam search

## [v0.6.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.6.0) (2017-04-07)

### New features

* Add new encoders: deep bidirectional and pyramidal deep bidirectional
* Add attention variants: no attention and *dot*, *general* or *concat* global attention
* Add alternative learning rate decay strategy for SGD training
* Introduce dynamic parameter change for dropout and fixed word embeddings
* Add length and coverage normalization during the beam search
* Add translation option to dump input sentence encoding
* Add TensorBoard metrics visualisation with [Crayon](https://github.com/torrvision/crayon)
* [*experimental*] Add sequence tagger model

### Fixes and improvements

* [*Breaking, changed option*] `-fix_word_vecs` options now accept `0` and `1` for a better retraining experience
* Check consistency of option settings when training from checkpoints
* Save and restore random number generator states from checkpoints
* Output more dataset metrics during the preprocessing
* Improve error message on invalid options
* Fix missing n-best hypotheses list in the output file
* Fix individual losses that were always computed when using random sampling
* Fix duplicated logs in parallel mode

## [v0.5.3](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.5.3) (2017-03-30)

### Fixes and improvements

* Fix data loading during training

## [v0.5.2](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.5.2) (2017-03-29)

### Fixes and improvements

* Improve compatibility with older Torch versions missing the `fmod` implementation

## [v0.5.1](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.5.1) (2017-03-28)

### Fixes and improvements

* Fix translation with FP16 precision
* Fix regression that make `tds` mandatory for translation

## [v0.5.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.5.0) (2017-03-06)

### New features

* Training code is now part of the library
* Add `-fallback_to_cpu` option to continue execution on CPU if GPU can't be used
* Add standalone script to generate vocabularies
* Add script to extract word embeddings
* Add option to prune vocabularies by minimum word frequency
* New REST server
* [*experimental*] Add data sampling during training
* [*experimental*] Add half floating point (fp16) support (with [cutorch@359ee80](https://github.com/torch/cutorch/commit/359ee80be391028ffa098de429cc0533b2f268f5))

### Fixes and improvements

* Make sure released model does not contain any serialized function
* Reduce size of released BRNN models (up to 2x smaller)
* Reported metrics are no longer averaged on the entire epoch
* Improve logging in asynchronous training
* Allow fixing word embeddings without providing pre-trained embeddings
* Fix pretrained word embeddings that were overriden by parameters initialization
* Fix error when using translation server with GPU model
* Fix gold data perplexity reporting during translation
* Fix wrong number of attention vectors returned by the translator

## [v0.4.1](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.4.1) (2017-02-16)

### Fixes and improvements

* Fix translation server error when clients send escaped unicode sequences
* Fix compatibility issue with the `:split()` function

## [v0.4.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.4.0) (2017-02-10)

### New features

* Profiler option
* Support hypotheses filtering during the beam search
* Support individually setting features vocabulary and embedding size
* [*experimental*] Scripts to interact with the [benchmark platform](http://scorer.nmt-benchmark.net/)
* [*experimental*] Language modeling example

### Fixes and improvements

* [*Breaking, new API*] Improve translator API consistency
* Improve beam search speed (up to 90% faster)
* Reduce released model size (up to 2x smaller)
* Fix tokenization of text containing the joiner marker character
* Fix `-joiner_new` option when using BPE
* Fix joiner marker generated without the option enabled
* Fix translation server crash on Lua errors
* Fix error when loading configuration files containing the `gpuid` option
* Fix BLEU drop when applying beam search on some models
* Fix error when using asynchronous parallel mode
* Fix non SGD model serialization after retraining
* Fix error when using `-replace_unk` with empty sentences in the batch
* Fix error when translating empty batch

## [v0.3.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.3.0) (2017-01-23)

### New features

* ZeroMQ translation server
* Advanced log management
* GRU cell
* Tokenization option to make the token separator an independent token
* Tokenization can run in parallel mode

### Fixes and improvements

* [*Breaking, renamed option*] Rename `-epochs` option to `-end_epoch` to clarify its behavior
* [*Breaking, removed option*] Remove `-nparallel` option and support a list of comma-separated identifiers on `-gpuid`
* [*Breaking, renamed option*] Zero-Width Joiner unicode character (ZWJ) is now tokenizing - but as a joiner
* Fix Hangul tokenization
* Fix duplicated tokens in aggressive tokenization
* Fix error when using BRNN and multiple source features
* Fix error when preprocessing empty lines and using additional features
* Fix error when translating empty sentences
* Fix error when retraining a BRNN model on multiple GPUs

## [v0.2.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.2.0) (2017-01-02)

### New features

* [*Breaking, renamed option*] Control maximum source and target length independently
* Asynchronous SGD
* Detokenization
* BPE support in tokenization

### Fixes and improvements

* Smaller memory footprint during training
* Smaller released model size after a non-SGD training
* Fix out of memory errors in preprocessing
* Fix BRNN models serialization and release
* Fix error when retraining a model
* Fix error when using more than one feature

## [v0.1.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.1.0) (2016-12-19)

Initial release.
