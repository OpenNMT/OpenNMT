The default translation mode allows the model to produce the `<unk>` symbol when it is not sure of the specific target word.

Often times `<unk>` symbols will correspond to proper names that can be directly transposed between languages. The `-replace_unk` option will substitute `<unk>` with source words that have the highest attention weight.

## Phrase table

Alternatively, advanced users may prefer to provide a preconstructed phrase table from an external aligner (such as `fast_align`) using the `-phrase_table` option to allow for non-identity replacement.

Instead of copying the source token with the highest attention, it will lookup in the phrase table for a possible translation. If a valid replacement is not found only then the source token will be copied.

The phrase table is a file with one translation per line in the format:

```text
source|||target
```

Where `source` and `target` are **case sensitive** and **single** tokens.

## Workarounds

Several techniques exist to minimize the out-of-vocabulary issue. This includes:

* subtokenization like BPE or wordpiece
* mixed word/characters model as desribed in [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf)
