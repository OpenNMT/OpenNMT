## [Unreleased]

### New features

* ZeroMQ translation server
* Advanced log management
* Tokenization option to make the token separator an independent token

### Fixes and improvements

* Fix Hangul tokenization
* Fix duplicated tokens in aggressive tokenization
* Fix error when using BRNN and multiple source features
* Fix error when preprocessing empty lines and using additional features

## [v0.2.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.2.0) (2017-01-02)

### New features

* Asynchronous SGD
* Detokenization
* BPE support in tokenization
* Control maximum source and target length independently

### Fixes and improvements

* Smaller memory footprint during training
* Smaller released model size after a non-SGD training
* Fix out of memory errors in preprocessing
* Fix BRNN models serialization and release
* Fix error when retraining a model
* Fix error when using more than one feature

## [v0.1.0](https://github.com/OpenNMT/OpenNMT/releases/tag/v0.1.0) (2016-12-19)

Initial release.
