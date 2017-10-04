<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`learn_bpe.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## BPE options

* `-size <string>` (default: `30000`)<br/>The number of merge operations to learn.
* `-bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. `prefix`: append `<w>` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `</w>` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-save_bpe <string>` (required)<br/>Path to save the output model.

## Tokenizer options

* `-tok_mode <string>` (accepted: `conservative`, `aggressive`, `space`; default: `space`)<br/>Define how aggressive should the tokenization be. `space` is space-tokenization.
* `-tok_joiner_annotate [<boolean>]` (default: `false`)<br/>Include joiner annotation using `-joiner` character.
* `-tok_joiner <string>` (default: `￭`)<br/>Character used to annotate joiners.
* `-tok_joiner_new [<boolean>]` (default: `false`)<br/>In `-joiner_annotate` mode, `-joiner` is an independent token.
* `-tok_case_feature [<boolean>]` (default: `false`)<br/>Generate case feature.
* `-tok_segment_case [<boolean>]` (default: `false`)<br/>Segment case feature, splits AbC to Ab C to be able to restore case
* `-tok_segment_alphabet <table>` (accepted: `Tagbanwa`, `Gurmukhi`, `Thaana`, `Cyrillic`, `Mongolian`, `Hanunoo`, `Han`, `Sinhala`, `Greek`, `Kanbun`, `Devanagari`, `Bengali`, `Lao`, `Syriac`, `Tamil`, `Limbu`, `Khmer`, `Kangxi`, `Tibetan`, `Arabic`, `Ethiopic`, `Malayalam`, `Tagalog`, `Ogham`, `Hangul`, `Georgian`, `Myanmar`, `Braille`, `Yi`, `Buhid`, `Thai`, `Kannada`, `Telugu`, `Katakana`, `Armenian`, `Latin`, `Cherokee`, `Hiragana`, `Hebrew`, `Bopomofo`, `Gujarati`, `Oriya`)<br/>Segment all letters from indicated alphabet.
* `-tok_segment_numbers [<boolean>]` (default: `false`)<br/>Segment numbers into single digits.
* `-tok_segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-tok_normalize_cmd <string>` (default: `''`)<br/>Command for on-the-fly corpus normalization. It should work in 'pipeline' mode.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NOERROR`; default: `INFO`)<br/>Output logs at this level and above.
