Data sampling is a technique to select a subset of the training set at each epoch. This could be a way to make the epoch unit smaller or select relevant training sequences at each epoch. There are different types of sampling that are selected using `-sample_type` option as defined below.

When sampling, with the option `-sample_vocab` it is also possible to restrict the generated vocabulary to the current sample which gives an approximate of the full softmax as defined here [Jean et al, 2015](http://www.aclweb.org/anthology/P15-1001) via an "importance sampling" approach.

!!! tip "Tip"
    Importance sampling is particularly useful when training systems with very large output vocabulary for faster computation.

## Uniform

The simplest data sampling is to uniformly select a subset of the training data. Using the `-sample N` option, the training will randomly choose \(N\) training sequences at each epoch.

A typical use case is to reduce the length of the epochs for more frequent learning rate updates and validation perplexity computation.

## Perplexity-based

This approach is an attempt to feed relevant training data at each epoch. When using the flag `-sample_type perplexity`, the perplexity of each sequence is used to generate a multinomial probability distribution over the training sequences. The higher the perplexity, the more likely the sequence is selected.

Alternatively, perplexity-based sampling can be enabled when an average training perplexity is met with the `-sample_perplexity_init` option.

!!! warning "Warning"
    This perplexity-based approach is experimental and effects are to be experimented. This also results in a ~10% slowdown as the perplexity of **each** sequence has to be independently computed.

## Partition

When using the flag `-sample_type partition`, samples are drawn without random, uniformally and incrementally from the corpus training. Use this mode for making sure all training sequences will be sent the same number of time.
