# Tools

This directory contains additional tools.

## Tokenization/Detokenization

### Dependencies

* `bit32` for Lua < 5.2

### Tokenization
To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-mode`: can be `aggressive` or `conservative` (default). In conservative mode, letters, numbers and '_' are kept in sequence, hyphens are accepted as part of tokens. Finally inner characters `[.,]` are also accepted (url, numbers).
* `-sep_annotate`: if set, add reversible separator mark to indicate separator-less or BPE tokenization (preference on symbol, then number, then letter)
* `-case_feature`: generate case feature - and convert all tokens to lowercase
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercased (opennmt)
  * `U`: token is uppercased (OPENNMT)
  * `C`: token is capitalized (Opennmt)
  * `M`: token case is mixed (OpenNMT)
* `-bpe_model`: when set, activate BPE using the BPE model filename

Note:

* `￨` is the feature separator symbol - if such character is used in source text, it is replace by its non presentation form `│`
* `￭` is the separator mark (generated in `-sep_annotate marker` mode) - if such character is used in source text, it is replace by its non presentation form `■`

### Detokenization

If you activate `sep_annotate` marker, the tokenization is reversible - just use:

```
th tools/detokenize.lua [-case_feature] < file.tok > file.detok
```

## Release model

After training a model on the GPU, you may want to release it to run on the CPU with the `release_model.lua` script.

```
th tools/release_model.lua -model model.t7 -gpuid 1
```

By default, it will create a `model_release.t7` file. See `th tools/release_model.lua -h` for advanced options.
