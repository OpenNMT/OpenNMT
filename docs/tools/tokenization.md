OpenNMT provides generic tokenization utilities to quickly process new training data. The goal of the tokenization is to convert raw sentences into sequences of tokens. In that process two main operations are performed in sequence:

* normalization - which applies some uniform transformation on the source sequences to identify and protect some specific sequences (for instance url), normalize characters (for instance all types of quotes, unicode variants) or even to normalize some variants (like dates) into unique representation simpler for the translation process
* the tokenization itself - which transform the actual normalized sentence into a sequence of space-separated tokens together with possible features (case).

## Normalization

Normalization is performed by user commandline tool which has to work in "pipeline" mode: sentences from standard input are normalized and produced on the standard output. For instance, the following python script is normalizing unicode representation (using NFC representation), turns French quotes `«»` into English quotes `“”`, and protect "hashtags" sequences:

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

import unicodedata
import re
import sys

for line in sys.stdin:
  line = line.strip()
  line = unicodedata.normalize('NFC', line.encode('utf8'))
  line = line.replace(u"«", u"“").replace(u"»", u"”")
  line = line.encode('utf8').sub(r'(^|[^S\w])#([A-Za-z0-9_]+)', '\\1｟#\\2｠')
  print(line)
```

Normalization script is called as part of tokenization adding the option `-normalize_cmd "normalize.py"`.

## Tokenization

To tokenize a corpus:

```bash
th tools/tokenize.lua OPTIONS < file > file.tok
```

Available tokenization modes are defined [here](http://opennmt.net/OpenNMT/options/tokenize/).

In order to perform `detokenization`, tokenization can introduce a joiner annotation mark `￭`.

## Detokenization

If you activate `-joiner_annotate` marker, the tokenization is reversible. Just use:

```bash
th tools/detokenize.lua OPTIONS < file.tok > file.detok
```

## Special characters

* `￨` (U+FFE8) is the feature separator symbol. If such character is used in source text, it is replaced by its non presentation form `│` (U+2502).
* `￭` (U+FFED) is the default joiner marker (generated in `-joiner_annotate marker` mode). If such character is used in source text, it is replaced by its non presentation form `■` (U+25A0)
* `｟...｠` (U+FF5F, U+FF60) are marking a sequence as protected - it won't be tokenized and its case feature is `N`.

## Mixed casing words
`-segment_case` feature enables tokenizer to segment words into subwords with one of 3 casing types (truecase ('House'), uppercase ('HOUSE') or lowercase ('house')), which helps  restore right casing during  detokenization. This feature is especially useful for texts with a signficant number of words with mixed casing ('WiFi' -> 'Wi' and 'Fi').

```text
WiFi --> wi￨C fi￨C
TVs --> tv￨U s￨L
```

## Alphabet Segmentation
Two options provide specific tokenization depending on alphabet:

* `-segment_alphabet_change`: tokenize a sequence between two letters when their alphabets differ - for instance between a Latin alphabet character and a Han character.
* `-segment_alphabet Alphabet`: tokenize all words of the indicated alphabet into characters - for instance to split a chinese sentence into characters, use `-segment_alphabet Han`:

```text
君子之心不胜其小，而气量涵盖一世。 --> 君 子 之 心 不 胜 其 小 ， 而 气 量 涵 盖 一 世 。
```

## Number Segmentation

The option `-segment_number` tokenizes numbers by digits. This option is interesting for full handling of numeric entities conversion/translation by neural networks.

```text
1984 --> 1 9 8 4
```


## BPE

OpenNMT's BPE module fully supports the [original BPE](https://github.com/rsennrich/subword-nmt) as default mode:

```bash
tools/learn_bpe.lua -size 30000 -save_bpe codes < input_tokenized
tools/tokenize.lua -bpe_model codes < input_tokenized
```

with three additional features:

**1\. Accept raw text as input and use OpenNMT's tokenizer for pre-tokenization before BPE training**

```bash
tools/learn_bpe.lua -size 30000 -save_bpe codes -tok_mode aggressive -tok_segment_alphabet_change [ OTHER_TOK_OPTIONS ] [ OTHER_BPE_TRAINING_OPTIONS ] < input_raw
tools/tokenize.lua -bpe_model codes -mode aggressive -segment_alphabet_change [ SAME_TOK_OPTIONS ] [ OTHER_BPE_INFERENCE_OPTIONS ] < input_raw
```

!!! note "Note"
    All TOK_OPTIONS for learn_bpe.lua have their equivalent for tokenize.lua without the prefix `tok_`
    BPE_INFERENCE_OPTIONS for tokenize.lua are those of Tokenizer options with the prefix `bpe_`

!!! warning "Warning"
    When applying BPE for any data set, the same TOK_OPTIONS should be used for learn_bpe.lua and tokenize.lua

**2\. Add BPE_TRAINING_OPTION for different modes of handling prefixes and/or suffixes: `-bpe_mode`**

* `suffix`: BPE merge operations are learnt to distinguish sub-tokens like "ent" in the middle of a word and "ent<\w>" at the end of a word. "<\w>" is an artificial marker appended to the end of each token input and treated as a single unit before doing statistics on bigrams. This is the default mode which is useful for most of the languages.
* `prefix`: BPE merge operations are learnt to distinguish sub-tokens like "ent" in the middle of a word and "<w\>ent" at the beginning of a word. "<w\>" is an artificial marker appended to the beginning of each token input and treated as a single unit before doing statistics on bigrams.
* `both`: `suffix` + `prefix`
* `none`: No artificial marker is appended to input tokens, a sub-token is treated equally whether it is in the middle or at the beginning or at the end of a token.

**3\. Add BPE_INFERENCE_OPTION for BPE in addition to the case feature: `-bpe_case_insensitive`**

OpenNMT's tokenization flow first applies BPE then add the case feature for each input token. With the standard BPE, "Constitution" and "constitution" may result in the different sequences of sub-tokens:

```text
Constitution --> con￨C sti￨l tu￨l tion￨l
constitution --> consti￨l tu￨l tion￨l
```

If you want a *caseless* split so that you can take the best from using case feature, and you can achieve that with the following command lines:

```bash
# We don't need BPE to care about case
tools/learn_bpe.lua -size 30000 -save_bpe codes_lc -tok_case_feature [ OTHER_TOK_OPTIONS ] [ OTHER_BPE_TRAINING_OPTIONS ] < input_raw

# The case information is preserved in the true case input
tools/tokenize.lua -bpe_model codes_lc -bpe_case_insensitive -case_feature [ SAME_TOK_OPTIONS ] [ OTHER_BPE_INFERENCE_OPTIONS ] < input_raw
```

The output of the previous example would be:

```text
Constitution --> con￨C sti￨l tu￨l tion￨l
constitution --> con￨l sti￨l tu￨l tion￨l
```
