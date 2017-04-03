<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`preprocess.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Read options from config file.
* `-save_config <string>`<br/>Save options from config file.

## Preprocess options

* `-data_type <string>`<br/>Type of data to preprocess. Use 'monotext' for monolingual data. This option impacts all options choices. (accepted: `bitext`, `monotext`; default: `bitext`)
* `-save_data <string>`<br/>Output file for the prepared data.

## Data options

* `-train_src <string>`<br/>Path to the training source data.
* `-train_tgt <string>`<br/>Path to the training target data.
* `-valid_src <string>`<br/>Path to the validation source data.
* `-valid_tgt <string>`<br/>Path to the validation target data.
* `-src_vocab <string>`<br/>Path to an existing source vocabulary.
* `-tgt_vocab <string>`<br/>Path to an existing target vocabulary.
* `-src_vocab_size <string>`<br/>Comma-separated list of source vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned. (default: `50000`)
* `-tgt_vocab_size <string>`<br/>Comma-separated list of target vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned. (default: `50000`)
* `-src_words_min_frequency <string>`<br/>Comma-separated list of source words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size. (default: `0`)
* `-tgt_words_min_frequency <string>`<br/>Comma-separated list of target words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size. (default: `0`)
* `-src_seq_length <number>`<br/>Maximum source sequence length. (default: `50`)
* `-tgt_seq_length <number>`<br/>Maximum target sequence length. (default: `50`)
* `-features_vocabs_prefix <string>`<br/>Path prefix to existing features vocabularies.
* `-time_shift_feature <number>`<br/>Time shift features on the decoder side. (default: `1`)
* `-sort <number>`<br/>If = 1, sort the sentences by size to build batches without source padding. (default: `1`)
* `-shuffle <number>`<br/>If = 1, shuffle data (prior sorting). (default: `1`)

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>`<br/>Output logs at this level and above. (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)

## Other options

* `-seed <number>`<br/>Random seed. (default: `3425`)
* `-report_every <number>`<br/>Report status every this many sentences. (default: `100000`)

