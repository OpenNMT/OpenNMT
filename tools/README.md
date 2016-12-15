# Tools

This directory contains additional tools

## Tokenization

To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-mode`: can be `aggressive` (default) or `conservative`. In conservative mode, letters, numbers and '_' are kept in sequence, hyphens are accepted as part of tokens. Finally inner characters `[.,]` are also accepted (url, numbers).
* `-sep_feature`: generate separator feature `S` means that the token is preceded by a space, `N` means that there is not space in original corpus
* `-case_feature`: indicate case of the token
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercased
  * `U`: token is uppercased
  * `C`: token is capitalized
  * `M`: token case is mixed

