OpenNMT provides generic tokenization utilities to quickly process new training data. For LuaJIT users, tokenization tools require the `bit32` package.

## Tokenization

To tokenize a corpus:

```bash
th tools/tokenize.lua OPTIONS < file > file.tok
```

See `th tools/tokenize.lua -h` for a detailled list of options.

## Detokenization

If you activate `-joiner_annotate` marker, the tokenization is reversible. Just use:

```bash
th tools/detokenize.lua OPTIONS < file.tok > file.detok
```

## Special characters

* `￨` is the feature separator symbol. If such character is used in source text, it is replaced by its non presentation form `│`.
* `￭` is the default joiner marker (generated in `-joiner_annotate marker` mode). If such character is used in source text, it is replaced by its non presentation form `■`

## BPE

Coming soon.
