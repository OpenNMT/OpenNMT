## Unknown words

The default translation mode allows the model to produce the UNK
symbol when it is not sure of the specific target word. Often times
UNK symbols will correspond to proper names that can be directly
transposed between languages. The `-replace_unk` option will
substitute UNK with a source word using the attention of the
model.

Alternatively, advanced users may prefer to provide a
preconstructed phrase table from an external aligner (such as
fast_align) using the `-phrase_table` option to allow for non-identity replacement.
Instead of copying the source token with the highest attention, it will
lookup in the phrase table for a possible translation. If a valid replacement
is not found then the source token will be copied.

The phrase table is a file with one translation per line in the format:

```
source|||target
```

Where `source` and `target` are **case sensitive** and **single** tokens.

## Inference models

After training a model, you may want to release it for inference only by using the `release_model.lua` script. A released model takes less space on disk and is compatible with CPU translation.

```
th tools/release_model.lua -model model.t7 -gpuid 1
```

By default, it will create a `model_release.t7` file. See `th tools/release_model.lua -h` for advanced options.
