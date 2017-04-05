The data prepration (or preprocessing) passes over the data to generate word vocabularies and sequences of indices used by the training.

## Delimiters

Training data are expected to follow the following format:

* sentences are newline-separated
* tokens are space-separated

## Vocabularies

The main goal of the preprocessing is to build the word vocabularies and assign each word to an index within these dictionaries. By default, word vocabularies are limited to 50,000. You can change this value with the `-src_vocab_size` and `-tgt_vocab_size`. Alternatively, you can prune the vocabulary size by setting the minimum frequency of words with the `-src_words_min_frequency` and `-tgt_words_min_frequency` options. The preprocessing script will generate `*.dict` files containing the vocabularies.

These files are optional for the rest of the workflow. However, it is common to reuse vocabularies across dataset using the `-src_vocab` and `-tgt_vocab` options. This is particularly needed when retraining a model on new data: the vocabulary has to be the same.

Vocabularies can be generated beforhand with the `tools/build_vocab.lua` script.

!!! note "Note"
    When pruning vocabularies to 50,000, the preprocessing will actually report a vocabulary size of 50,004 because of 4 special tokens that are automatically added.

## Shuffling and sorting

By default, OpenNMT both shuffles and sorts the data before the training. This process comes from 2 constraints of batch training:

* **shuffling**: sentences within a batch should come from different parts of the corpus
* **sorting**: sentences within a batch should have the same source length (i.e. without padding to maximize efficiency)

!!! note "Note"
    During the training, batches are also randomly selected unless the `-curriculum` option is used.
