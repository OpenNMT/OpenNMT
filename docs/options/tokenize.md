<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`tokenize.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Tokenizer options

* `-mode <string>` (accepted: `space`, `conservative`, `aggressive`; default: `conservative`)<br/>Define how aggressive should the tokenization be. `aggressive` only keeps sequences of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65", "soft-landing", etc. `space` is doing space tokenization.
* `-joiner_annotate [<boolean>]` (default: `false`)<br/>Include joiner annotation using `-joiner` character.
* `-joiner <string>` (default: `￭`)<br/>Character used to annotate joiners.
* `-joiner_new [<boolean>]` (default: `false`)<br/>In `-joiner_annotate` mode, `-joiner` is an independent token.
* `-case_feature [<boolean>]` (default: `false`)<br/>Generate case feature.
* `-segment_case [<boolean>]` (default: `false`)<br/>Segment case feature, splits AbC to Ab C to be able to restore case
* `-segment_alphabet <table>` (accepted: `Tagalog`, `Hanunoo`, `Limbu`, `Yi`, `Hebrew`, `Latin`, `Devanagari`, `Thaana`, `Lao`, `Sinhala`, `Georgian`, `Kannada`, `Cherokee`, `Kanbun`, `Buhid`, `Malayalam`, `Han`, `Thai`, `Katakana`, `Telugu`, `Greek`, `Myanmar`, `Armenian`, `Hangul`, `Cyrillic`, `Ethiopic`, `Tagbanwa`, `Gurmukhi`, `Ogham`, `Khmer`, `Arabic`, `Oriya`, `Hiragana`, `Mongolian`, `Kangxi`, `Syriac`, `Gujarati`, `Braille`, `Bengali`, `Tamil`, `Bopomofo`, `Tibetan`)<br/>Segment all letters from indicated alphabet.
* `-segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-bpe_model <string>` (default: `''`)<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-bpe_EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the End of Token while applying BPE in mode 'prefix' or 'both'.
* `-bpe_BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the Beginning of Token while applying BPE in mode 'suffix' or 'both'.
* `-bpe_case_insensitive [<boolean>]` (default: `false`)<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-bpe_BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-bpe_EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.

## Other options

* `-nparallel <number>` (default: `1`)<br/>Number of parallel thread to run the tokenization
* `-batchsize <number>` (default: `1000`)<br/>Size of each parallel batch - you should not change except if low memory
