<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`preprocess.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Preprocess options

* `-data_type <string>` (accepted: `bitext`, `monotext`, `feattext`; default: `bitext`)<br/>Type of data to preprocess. Use 'monotext' for monolingual data. This option impacts all options choices.
* `-save_data <string>` (required)<br/>Output file for the prepared data.
* `-check_plength [<boolean>]` (default: `false`)<br/>Check source and target have same length (for seq tagging).

## Data options

* `-train_src <string>` (required)<br/>Path to the training source data.
* `-train_tgt <string>` (required)<br/>Path to the training target data.
* `-valid_src <string>` (required)<br/>Path to the validation source data.
* `-valid_tgt <string>` (required)<br/>Path to the validation target data.
* `-src_vocab <string>` (default: `''`)<br/>Path to an existing source vocabulary.
* `-src_vocab_size <table>` (default: `50000`)<br/>List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are not pruned.
* `-src_words_min_frequency <table>` (default: `0`)<br/>List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are pruned by size.
* `-tgt_vocab <string>` (default: `''`)<br/>Path to an existing target vocabulary.
* `-tgt_vocab_size <table>` (default: `50000`)<br/>List of target vocabularies size: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are not pruned.
* `-tgt_words_min_frequency <table>` (default: `0`)<br/>List of target words min frequency: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are pruned by size.
* `-src_seq_length <number>` (default: `50`)<br/>Maximum source sequence length.
* `-tgt_seq_length <number>` (default: `50`)<br/>Maximum target sequence length.
* `-features_vocabs_prefix <string>` (default: `''`)<br/>Path prefix to existing features vocabularies.
* `-time_shift_feature [<boolean>]` (default: `true`)<br/>Time shift features on the decoder side.
* `-keep_frequency [<boolean>]` (default: `false`)<br/>Keep frequency of words in dictionary.
* `-sort [<boolean>]` (default: `true`)<br/>If set, sort the sequences by size to build batches without source padding.
* `-shuffle [<boolean>]` (default: `true`)<br/>If set, shuffle the data (prior sorting).
* `-idx_files [<boolean>]` (default: `false`)<br/>If set, source and target files are 'key value' with key match between source and target.
* `-report_every <number>` (default: `100000`)<br/>Report status every this many sentences.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-seed <number>` (default: `3425`)<br/>Random seed.
