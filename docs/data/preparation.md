The data preparation (or preprocessing) passes over the data to generate word vocabularies and sequences of indices used by the training.

## Data Type

By default, data type is `bitext` which are aligned source and target files - alignment is by default done at the line level, but can also be done through aligned index (see Index File type).
For training language models, data type is `monotext` which is only one language file.

Finally, you can also manipulate `feattext` data type (see below) which is allowing to code sequences of vectors (for instance for devices generating sequence of features in sequence).

!!! note "Note"
    Input Vectors can only be used for the source.


## Delimiters

Training data (for `bitext` and `monotext` data types) are expected to follow the following format:

* sentences are newline-separated
* tokens are space-separated

## Index Files

Index files are aligning different files by index and not by line. For instance the following files are aligned by index:

```
line1 First line
line2 Second line
```

```
line2 Deuxième ligne
line1 Première ligne
```

where the first token of each line is an index which must have an equivalent (at any position) in aligned files.

The option `-idx_files` is used (in `preprocess.lua` or `translate.lua`) to enable this feature.

## Input Vectors

OpenNMT supports use of vector sequence instead of word sequence for source.

The data type is `feattext` and is using [Kaldi](http://kaldi-asr.org) text ark dump format. For instance the following entry, indexed by `KEY` is representing a sequence
of `m` vectors of `n` values:

```
KEY [
FEAT1.1 FEAT1.2 FEAT1.3 ... FEAT1.n
...
FEATm.1 FEATm.2 FEATm.3 ... FEATm.n ]
```

!!! warning "Warning"
    Note that you need to use Index Files for representing Input Vectors.

## Vocabularies

The main goal of the preprocessing is to build the word vocabularies and assign each word to an index within these dictionaries. By default, word vocabularies are limited to 50,000. You can change this value with the `-src_vocab_size` and `-tgt_vocab_size`. Alternatively, you can prune the vocabulary size by setting the minimum frequency of words with the `-src_words_min_frequency` and `-tgt_words_min_frequency` options. The preprocessing script will generate `*.dict` files containing the vocabularies.

These files are optional for the rest of the workflow. However, it is common to reuse vocabularies across dataset using the `-src_vocab` and `-tgt_vocab` options. This is particularly needed when retraining a model on new data: the vocabulary has to be the same.

Vocabularies can be generated beforehand with the `tools/build_vocab.lua` script.

!!! note "Note"
    When pruning vocabularies to 50,000, the preprocessing will actually report a vocabulary size of 50,004 because of 4 special tokens that are automatically added.

## Shuffling and sorting

By default, OpenNMT both shuffles and sorts the data before the training. This process comes from 2 constraints of batch training:

* **shuffling**: sentences within a batch should come from different parts of the corpus
* **sorting**: sentences within a batch should have the same source length (i.e. without padding to maximize efficiency)

!!! note "Note"
    During the training, batches are also randomly selected unless the `-curriculum` option is used.
