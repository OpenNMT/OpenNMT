# Tools

This directory contains additional tools

## Tokenization

To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-mode`: can be `aggressive` or `conservative` (default). In conservative mode, letters, numbers and '_' are kept in sequence, hyphens are accepted as part of tokens. Finally inner characters `[.,]` are also accepted (url, numbers).
* `-sep_annotate`: indicate how to annotate non-separator tokenization - can be `marker` (default), `feature` or `none`:
  * `marker`: when a space is added for tokenization, add reversible -@- mark on one side (preference symbol, number, letter)
  * `feature`: generate separator feature `S` means that the token is preceded by a space, `N` means that there is not space in original corpus
  * `none`: don't annotate
* `-case_feature`: indicate case of the token
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercased
  * `U`: token is uppercased
  * `C`: token is capitalized
  * `M`: token case is mixed

