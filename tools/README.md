This directory contains additional tools and extensions to OpenNMT.

* `apply_embeddings.lua` applies pre-trained word embeddings to a text file and generate an vector file.
* `average_models.lua` averages all parameters of multiple models.
* `build_vocab.lua` generates vocabulary files from text.
* `detokenize.lua` runs OpenNMT's generic detokenization on a text file.
* `embeddings.lua` generates pre-trained word embeddings.
* `extract_embeddings.lua` outputs word embeddings from a model to a text file.
* `learn_bpe.lua` learns a BPE model.
* `release_model.lua` releases trained model for inference.
* `rest_translation_server.lua` runs a simple translation server with a REST API.
* `score.lua` scores predictions against references with a metric.
* `tokenize.lua` runs OpenNMT's generic tokenization on a text file.
* `translation_server.lua` runs a simple ZMQ-based translation server.

For more details about these scripts, visit the [online documentation](http://opennmt.net/OpenNMT/) or explore available options with the `-h` flag.
