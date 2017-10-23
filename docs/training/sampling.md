Data sampling is a technique to select a subset of the training set at each epoch. This could be a way to make the epoch unit smaller or select relevant training sequences at each epoch. This is also necessary for working on very large dataset - where the full data does not need to be loaded in memory for each epoch. There are two implementations for sampling: _memory sampling_ and _file sampling_.

Both implementations also support vocabulary sampling (also called _Importance Sampling_).

## Importance Sampling

When sampling, with the option `-sample_vocab` it is also possible to restrict the generated vocabulary to the current sample which gives an approximate of the full softmax as defined here [Jean et al, 2015](http://www.aclweb.org/anthology/P15-1001) via an "importance sampling" approach.

!!! tip "Tip"
    Importance sampling is particularly useful when training systems with very large output vocabulary for faster computation.

## Memory Sampling

_Memory Sampling_ is enabled using `-sample N` option where \(N\) is the number of sequences to select at each epoch. There are different methods for selecting these \(N\) sentences corresponding to the `-sample_type` option: `uniform` (default), `perplexity` or `partition`.

### Uniform

The simplest data sampling is to uniformly select a subset of the training data. Using the `-sample N` option, the training will randomly choose \(N\) training sequences at each epoch.

A typical use case is to reduce the length of the epochs for more frequent learning rate updates and validation perplexity computation.

### Perplexity-based

This approach is an attempt to feed relevant training data at each epoch. When using the flag `-sample_type perplexity`, the perplexity of each sequence is used to generate a multinomial probability distribution over the training sequences. The higher the perplexity, the more likely the sequence is selected.

Alternatively, perplexity-based sampling can be enabled when an average training perplexity is met with the `-sample_perplexity_init` option.

!!! warning "Warning"
    This perplexity-based approach is experimental and effects are to be experimented. This also results in a ~10% slowdown as the perplexity of **each** sequence has to be independently computed.

### Partition

When using the flag `-sample_type partition`, samples are drawn without random, uniformally and incrementally from the corpus training. Use this mode for making sure all training sequences will be sent the same number of time.

## File Sampling

_File Sampling_ is enabled using `-gsample V` option: \(V\) is either an integer and in that case it represents the number of sentences to sample from the dataset, or a float values and in that case, it represents a relative size based on the full dataset size (e.g: 0.1 being 10%).

File Sampling can only be used with on-the-fly preprocessing and tokenization as an alternative to sequential tokenization, preprocessing, training - and this is refers as _Dynamic Dataset_ below.

In _File Sampling_, the only available sampling method is uniform meaning that the sentences are selected uniformly in each corpus of the dataset. However, it is possible to modify the distribution of the sampling for the different files using sampling rule file as described below.

### Dynamic Dataset

It is possible to provide raw files directly to training script. For that, instead of using the `-data D` option, you have to use preprocessing data selection options (such as `-train_src`, `-train_tgt`, or `-training_dir` option). Note these modes are exclusive. Corpus can be pre-tokenized, or you can provide tokenization options for both source and target (or source only for language models) prefixing all tokenization options with `-tok_src_`, `-tok_tgt_` or `tok_`. For instance - the following commandlines use all the files from `baseline` directory with source `.en` suffix and target `.fr` suffix. The source is tokenized in aggressive mode, and is using case feature, while the target is tokenized in aggressive mode and limited to 30 words sequences.

```
th train.lua
  -train_dir baseline
  -src_suffix .en -tgt_suffix .fr
  -tok_src_mode aggressive -tok_src_case_feature
  -tok_tgt_mode aggressive
  -tgt_seq_length 30
  -save_model baseline
```

The available options are `preprocess.lua` options documented [here](http://opennmt.net/OpenNMT/options/preprocess/) and `tokenize.lua` options documented [here](http://opennmt.net/OpenNMT/options/tokenize/).

!!! tip "Tip"
    It is simpler (and faster) to use dynamic dataset associated with file sampling when working on corpus with more than 10 million sentences.

### Sampling distribution rules

When the set of training files is heterogenous, you can specify the proportion of each file using a distribution rule file specified with `-gsample_dist FILE` option.

The rule file is list of rule in each line applied. A rule is a `LuaPattern SPACE WEIGHT`. The first rule in the file matching (with `LuaPattern` a filename) is applied for the file. `LuaPattern` can be a lua regex (see [https://www.lua.org/pil/20.2.html](https://www.lua.org/pil/20.2.html)) or `*` matching everything.

For instance, let us say you have the following files in your `train_dir` directory:

```
generic.src, generic.tgt
IT1.src, IT1.tgt
IT2.src, IT2.tgt
MSDN.src, MSDN.tgt
colloquial.src, colloquial.tgt
news.src, news.tgt
```

and using the following rules:
```
IT,MSDN 20
colloquial 10
generic 65
* 5
```

The following rules apply:

* `generic 65` matches `generic.{src,tgt}`
* `IT,MSDN 20` matches `IT{1,2}.{src,tgt}` and `MSDN.{src,tgt}`
* `colloquial 10` matches `colloquial.{src,tgt}`
* `* 5` maches `news.{src,tgt}`

The weights are dynamically normalized to 1. Here we will make sure that 65% of the sample will be composed of sentences from `generic.{src,tgt}` and only 20% from `IT{1,2}.{src,tgt}` and `MSDN.{src,tgt}`. To build the sample, the sampling preparation algorithm might oversample some of the corpus if is too small.

!!! tip "Tip"
    If you want to use all sentences of some training files without sampling, use `*` as the weight value.

!!! warning "Warning"
    If one file could not be match by a rule, it would be completely excluded.

To test your distribution rules, it is possible to execute a dry run of the preprocessor:

```bash
th preprocess.lua -gsample_dist rules.txt -gsample 100000 -train_dir data/ -dry_run
```
