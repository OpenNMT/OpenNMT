Data sampling is a technique to select a subset of the training set at each epoch. This could be a way to make the epoch unit smaller or select relevant training sequences at each epoch.

## Uniform

The simplest data sampling is to uniformly select a subset of the training data. Using the `-sample N` option, the training will randomly choose `N` training sequences at each epoch.

A typical use case is to reduce the length of the epochs for more frequent learning rate updates and validation perplexity computation.

## Perplexity-based

This approach is an attempt to feed relevant training data at each epoch. When using the flag `-sample_w_ppl`, the perplexity of each sequence is used to generate a multinomial probability distribution over the training sequences. The higher the perplexity, the more likely the sequence is selected.

Alternatively, perplexity-based sampling can be enabled when an average training perplexity is met with the `-sample_w_ppl_init` option.

**Note:** this perplexity-based approach is experimental and effects are to be experimented. This also results in a ~10% slowdown as the perplexity of **each** sequence has to be independently computed.
