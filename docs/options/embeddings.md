<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`embeddings.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Data options

* `-dict_file <string>` (default: `''`)<br/>Path to outputted dict file from `preprocess.lua`.
* `-embed_file <string>` (default: `''`)<br/>Path to the embedding file. Ignored if `-lang` is used.
* `-save_data <string>` (default: `''`)<br/>Output file path/label.

## Embedding options

* `-lang <string>` (default: `''`)<br/>Wikipedia Language Code to autoload embeddings.
* `-embed_type <string>` (accepted: `word2vec`, `glove`; default: `word2vec`)<br/>Embeddings file origin. Ignored if `-lang` is used.
* `-normalize [<boolean>]` (default: `true`)<br/>Boolean to normalize the word vectors, or not.
* `-report_every <number>` (default: `100000`)<br/>Print stats every this many lines read from embedding file.

