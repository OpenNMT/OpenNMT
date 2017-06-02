OpenNMT provides generic tokenization utilities to quickly process new training data.

!!! note "Note"
    For LuaJIT users, tokenization tools require the `bit32` package.

## Tokenization

To tokenize a corpus:

```bash
th tools/tokenize.lua OPTIONS < file > file.tok
```

## Detokenization

If you activate `-joiner_annotate` marker, the tokenization is reversible. Just use:

```bash
th tools/detokenize.lua OPTIONS < file.tok > file.detok
```

## Special characters

* `￨` is the feature separator symbol. If such character is used in source text, it is replaced by its non presentation form `│`.
* `￭` is the default joiner marker (generated in `-joiner_annotate marker` mode). If such character is used in source text, it is replaced by its non presentation form `■`

## Mixed casing words
`-segment_case` feature enables tokenizer to segment words into subwords with one of 3 casing types (truecase ('House'), uppercase ('HOUSE') or lowercase ('house')), which helps  restore right casing during  detokenization. This feature is especially useful for texts with a signficant number of words with mixed casing ('WiFi' -> 'Wi' and 'Fi').
```text
WiFi --> wi￨C fi￨C
TVs --> tv￨U s￨L
```

## BPE

OpenNMT's BPE module fully supports the [original BPE](https://github.com/rsennrich/subword-nmt) as default mode:

```bash
tools/learn_bpe.lua -size 30000 -save_bpe codes < input
tools/tokenize.lua -bpe_model codes < input
```

with two additional features:

**1\. Add support for different modes of handling prefixes and/or suffixes: `-bpe_mode`**

* `suffix`: BPE merge operations are learnt to distinguish sub-tokens like "ent" in the middle of a word and "ent<\w>" at the end of a word. "<\w>" is an artificial marker appended to the end of each token input and treated as a single unit before doing statistics on bigrams. This is the default mode which is useful for most of the languages.
* `prefix`: BPE merge operations are learnt to distinguish sub-tokens like "ent" in the middle of a word and "<w\>ent" at the beginning of a word. "<w\>" is an artificial marker appended to the beginning of each token input and treated as a single unit before doing statistics on bigrams.
* `both`: `suffix` + `prefix`
* `none`: No artificial marker is appended to input tokens, a sub-token is treated equally whether it is in the middle or at the beginning or at the end of a token.

**2\. Add support for BPE in addition to the case feature: `-bpe_case_insensitive`**

OpenNMT's tokenization flow first applies BPE then add the case feature for each input token. With the standard BPE, "Constitution" and "constitution" may result in the different sequences of sub-tokens:

```text
Constitution --> con￨C sti￨l tu￨l tion￨l
constitution --> consti￨l tu￨l tion￨l
```

If you want a *caseless* split so that you can take the best from using case feature, and you can achieve that with the following command lines:

```bash
# We don't need BPE to care about case
tools/learn_bpe.lua -size 30000 -save_bpe codes_lc < input_lowercased

# The case information is preserved in the true case input
tools/tokenize.lua -bpe_model codes_lc -bpe_case_insensitive < input
```

The output of the previous example would be:

```text
Constitution --> con￨C sti￨l tu￨l tion￨l
constitution --> con￨l sti￨l tu￨l tion￨l
```

!!! note "Note"
    Use Lua 5.2 if you encounter any memory issue while using `learn_bpe.lua` (e.g. `-size` is too big). Otherwise, stay with Lua 5.1 for better efficiency.
