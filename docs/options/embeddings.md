<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`embeddings.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Data options

* `-dict_file <string>` (required)<br/>Path to outputted dict file from `preprocess.lua`.
* `-embed_file <string>` (default: `''`)<br/>Path to the embedding file. Ignored if `-lang` is used.
* `-save_data <string>` (required)<br/>Output file path/label.
* `-save_unknown_dict <string>` (default: `''`)<br/>Path to file for saving vocabs not found in embedding.

## Embedding options

* `-lang <string>` (default: `''`)<br/>Wikipedia Language Code to autoload embeddings.
* `-embed_type <string>` (accepted: `word2vec-bin`, `word2vec-txt`, `glove`; default: `word2vec-bin`)<br/>Embeddings file origin. Ignored if `-lang` is used.
* `-normalize [<boolean>]` (default: `true`)<br/>Boolean to normalize the word vectors, or not.
* `-approximate [<boolean>]` (default: `false`)<br/>If set, will also look for variants (case, joiner annotate) to match dictionary and word embedding.
* `-report_every <number>` (default: `100000`)<br/>Print stats every this many lines read from embedding file.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout - if file name ending with json, output structure json.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NONE`; default: `INFO`)<br/>Output logs at this level and above.
