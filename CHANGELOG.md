## [Unreleased]

### New features

* Training code is now part of the library
* Add `-fallback_to_cpu` option to continue execution on CPU if GPU can't be used
* Add standalone script to generate vocabularies
* Add script to extract word embeddings
* Add option to prune vocabularies by minimum word frequency
* Add data sampling during training

### Fixes and improvements

* Make sure released model does not contain any serialized function
* Reported metrics are no longer averaged on the entire epoch
* Improve logging in asynchronous training
* Allow fixing word embeddings without providing pre-trained embeddings
* Fix error when using translation server with GPU model
* Reduce size of released BRNN models (up to 2x smaller)
* Fix gold data perplexity reporting during translation
* Fix pretrained word embeddings that were overriden by parameters initialization

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
