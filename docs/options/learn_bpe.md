<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`learn_bpe.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## BPE options

* `-size <string>` (default: `30000`)<br/>The number of merge operations to learn.
* `-lc [<boolean>]` (default: `false`)<br/>Lowercase input tokens before learning BPE.
* `-bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. `prefix`: append `<w>` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `</w>` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-save_bpe <string>` (required)<br/>Path to save the output model.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.
