<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`tokenize.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Load options from this file.
* `-save_config <string>`<br/>Save options to this file.

## Tokenizer options

* `-mode <string>` (accepted: `conservative`, `aggressive`; default: `conservative`)<br/>Define how aggressive should the tokenization be. `aggressive` only keeps sequences of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65", "soft-landing", etc.
* `-joiner_annotate`<br/>Include joiner annotation using `-joiner` character.
* `-joiner <string>` (default: `￭`)<br/>Character used to annotate joiners.
* `-joiner_new`<br/>In `-joiner_annotate` mode, `-joiner` is an independent token.
* `-case_feature`<br/>Generate case feature.
* `-bpe_model <string>`<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, `-mode` will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the end of token.
* `-BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the beginning of token.
* `-bpe_case_insensitive`<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.

## Other options

* `-nparallel <number>` (default: `1`)<br/>Number of parallel thread to run the tokenization
* `-batchsize <number>` (default: `1000`)<br/>Size of each parallel batch - you should not change except if low memory

