# preprocess.lua

* `-h`: This help.
* `-md`: Dump help in Markdown format.
* `-config`: Read options from config file.
* `-save_config`: Save options from config file.

## Preprocess options

* `-data_type`: Type of data to preprocess. Use 'monotext' for monolingual data. This option impacts all options choices. (accepted: `bitext`, `monotext`; default: `bitext`)
* `-save_data`: Output file for the prepared data.

## Data options

* `-train_src`: Path to the training source data.
* `-train_tgt`: Path to the training target data.
* `-valid_src`: Path to the validation source data.
* `-valid_tgt`: Path to the validation target data.
* `-src_vocab`: Path to an existing source vocabulary.
* `-tgt_vocab`: Path to an existing target vocabulary.
* `-src_vocab_size`: Comma-separated list of source vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned. (default: `50000`)
* `-tgt_vocab_size`: Comma-separated list of target vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned. (default: `50000`)
* `-src_words_min_frequency`: Comma-separated list of source words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size. (default: `0`)
* `-tgt_words_min_frequency`: Comma-separated list of target words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size. (default: `0`)
* `-src_seq_length`: Maximum source sequence length. (default: `50`)
* `-tgt_seq_length`: Maximum target sequence length. (default: `50`)
* `-features_vocabs_prefix`: Path prefix to existing features vocabularies.
* `-time_shift_feature`: Time shift features on the decoder side. (default: `1`)
* `-sort`: If = 1, sort the sentences by size to build batches without source padding. (default: `1`)
* `-shuffle`: If = 1, shuffle data (prior sorting). (default: `1`)

## Logger options

* `-log_file`: Output logs to a file under this path instead of stdout.
* `-disable_logs`: If set, output nothing.
* `-log_level`: Output logs at this level and above. (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)

## Other options

* `-seed`: Random seed. (default: `3425`)
* `-report_every`: Report status every this many sentences. (default: `100000`)

