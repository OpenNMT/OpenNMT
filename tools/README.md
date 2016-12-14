# Tools

This directory contains additional tools

## Tokenization

To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-sep_feature`: generate separator feature `S` means that the token is preceded by a space, `N` means that there is not space in original corpus
* `-case_feature`: indicate case of the token
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercase
  * `U`: token is uppercas
  * `C`: token is capitalized
  * `M`: token case is mixed

