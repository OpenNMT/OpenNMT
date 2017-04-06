<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`learn_bpe.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Load options from this file.
* `-save_config <string>`<br/>Save options to this file.

## BPE options

* `-size <string>` (default: `30000`)<br/>The number of merge operations to learn.
* `-t`<br/>Tokenize the input with tokenizer, the same options as tokenize.lua, but only `-mode` is taken into account for BPE training.
* `-mode <string>` (accepted: `conservative`, `aggressive`; default: `conservative`)<br/>Define how aggressive should the tokenization be. `aggressive` only keeps sequences of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65", "soft-landing", etc.
* `-lc`<br/>Lowercase the output from the tokenizer before learning BPE.
* `-bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. `prefix`: append `<w>` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `</w>` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-save_bpe <string>`<br/>Path to save the output model.

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

