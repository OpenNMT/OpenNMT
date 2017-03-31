OpenNMT provides generic tokenization utilies to quickly process new training data. For LuaJIT users, tokenization tools require the `bit32` package.

## Tokenization

To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

See `th tools/tokenize.lua -h` for a detailled list of options.

## Detokenization

If you activate `-joiner_annotate` marker, the tokenization is reversible. Just use:

```
th tools/detokenize.lua OPTIONS < file.tok > file.detok
```

## Special characters

* `￨` is the feature separator symbol - if such character is used in source text, it is replace by its non presentation form `│`
* `￭` is the default joiner marker (generated in `-joiner_annotate marker` mode) - if such character is used in source text, it is replace by its non presentation form `■`

## BPE

TODO
