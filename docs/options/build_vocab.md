<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`build_vocab.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Vocabulary options

* `-data <string>` (required)<br/>Data file.
* `-save_vocab <string>` (required)<br/>Vocabulary dictionary prefix.
* `-vocab_size <table>` (default: `50000`)<br/>List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are not pruned.
* `-words_min_frequency <table>` (default: `0`)<br/>List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are pruned by size.
* `-keep_frequency [<boolean>]` (default: `false`)<br/>Keep frequency of words in dictionary.
* `-idx_files [<boolean>]` (default: `false`)<br/>If set, each line of the data file starts with a first field which is the index of the sentence.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

