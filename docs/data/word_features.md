OpenNMT supports additional features on source and target words in the form of **discrete labels**.

* On the source side, these features act as **additional information** to the encoder. An
embedding will be optimized for each label and then fed as additional source input
alongside the word it annotates.
* On the target side, these features will be **predicted** by the network. The
decoder is then able to decode a sentence and annotate each word.

To use additional features, directly modify your data by appending labels to each word with
the special character `￨` (unicode character FFE8). There can be an **arbitrary number** of additional
features in the form `word￨feat1￨feat2￨...￨featN` but each word must have the same number of
features and in the same order. Source and target data can have a different number of additional features.

As an example, see `data/src-train-case.txt` which uses a separate feature
to represent the case of each word. Using case as a feature is a way to optimize the word
dictionary (no duplicated words like "the" and "The") and gives the system an additional
information that can be useful to optimize its objective function.

```text
it￨C is￨l not￨l acceptable￨l that￨l ,￨n with￨l the￨l help￨l of￨l the￨l national￨l bureaucracies￨l ,￨n parliament￨C &apos;s￨l legislative￨l prerogative￨l should￨l be￨l made￨l null￨l and￨l void￨l by￨l means￨l of￨l implementing￨l provisions￨l whose￨l content￨l ,￨n purpose￨l and￨l extent￨l are￨l not￨l laid￨l down￨l in￨l advance￨l .￨n
```

You can generate this case feature with OpenNMT's tokenization script and the `-case_feature` flag.

## Time-shifting

By default, word features on the target side are automatically shifted compared to the words so that their prediction directly depends on the word they annotate. More precisely at timestep \(t\):

* the inputs are \(words^{(t)}\) and \(features^{(t-1)}\)
* the outputs are \(words^{(t+1)}\) and \(features^{(t)}\)

To reuse available vocabulary, \(features^{(-1)}\) is set to the end of sentence token.

## Vocabularies

By default, features vocabulary size is unlimited. Depending on the type of features you are using, you may want to limit their vocabulary during the preprocessing with the `-src_vocab_size` and `-tgt_vocab_size` options in the format `word_vocab_size[,feat1_vocab_size[,feat2_vocab_size[...]]]`. For example:

```bash
# unlimited source features vocabulary size
-src_vocab_size 50000

# first feature vocabulary is limited to 60, others are unlimited
-src_vocab_size 50000,60

# second feature vocabulary is limited to 100, others are unlimited
-src_vocab_size 50000,0,100

# limit vocabulary size of the first and second feature
-src_vocab_size 50000,60,100
```

You can similarly use `-src_words_min_frequency` and `-tgt_words_min_frequency` to limit vocabulary by frequency instead of absolute size.

Like words, word features vocabularies can be reused across datasets with the `-features_vocabs_prefix`. For example, if the processing generates theses features dictionaries:

* `data/demo.source_feature_1.dict`
* `data/demo.source_feature_2.dict`
* `data/demo.source_feature_3.dict`

you have to set `-features_vocabs_prefix data/demo` as command line option.

## Embeddings

The feature embedding size is automatically computed based on the number of values the feature takes. This default size reduction works well for features with few values like the case or POS.

For other features, you may want to manually choose the embedding size with the `-src_word_vec_size` and `-tgt_word_vec_size` options. They behave similarly to `-src_vocab_size` with a comma-separated list of embedding size: `word_vec_size[,feat1_vec_size[,feat2_vec_size[...]]]`.

Then, each feature embedding is concatenated to each other by default. You can instead choose to sum them by setting `-feat_merge sum`. Finally, the resulting merged embedding is concatenated to the word embedding.

!!! warning "Warning"
    In the `sum` case, each feature embedding must have the same dimension. You can set the common embedding size with `-feat_vec_size`.

## Beam search

During decoding, the beam search is only applied on the target words space and not on the word features. When the beam path is complete, the associated features are selected along this path.
