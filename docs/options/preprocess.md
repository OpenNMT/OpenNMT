<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`preprocess.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Preprocess options

* `-data_type <string>` (accepted: `bitext`, `monotext`, `feattext`; default: `bitext`)<br/>Type of data to preprocess. Use 'monotext' for monolingual data. This option impacts all options choices.
* `-save_data <string>` (required)<br/>Output file for the prepared data.

## Data options

* `-train_dir <string>` (default: `''`)<br/>Path to training files directory.
* `-train_src <string>` (default: `''`)<br/>Path to the training source data.
* `-train_tgt <string>` (default: `''`)<br/>Path to the training target data.
* `-valid_src <string>` (default: `''`)<br/>Path to the validation source data.
* `-valid_tgt <string>` (default: `''`)<br/>Path to the validation target data.
* `-src_vocab <string>` (default: `''`)<br/>Path to an existing source vocabulary.
* `-src_suffix <string>` (default: `.src`)<br/>Suffix for source files in train/valid directories.
* `-src_vocab_size <table>` (default: `50000`)<br/>List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are not pruned.
* `-src_words_min_frequency <table>` (default: `0`)<br/>List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are pruned by size.
* `-tgt_vocab <string>` (default: `''`)<br/>Path to an existing target vocabulary.
* `-tgt_suffix <string>` (default: `.tgt`)<br/>Suffix for target files in train/valid directories.
* `-tgt_vocab_size <table>` (default: `50000`)<br/>List of target vocabularies size: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are not pruned.
* `-tgt_words_min_frequency <table>` (default: `0`)<br/>List of target words min frequency: `word[ feat1[ feat2[ ...] ] ]`. If = 0, vocabularies are pruned by size.
* `-src_seq_length <number>` (default: `50`)<br/>Maximum source sequence length.
* `-tgt_seq_length <number>` (default: `50`)<br/>Maximum target sequence length.
* `-check_plength [<boolean>]` (default: `false`)<br/>Check source and target have same length (for seq tagging).
* `-features_vocabs_prefix <string>` (default: `''`)<br/>Path prefix to existing features vocabularies.
* `-time_shift_feature [<boolean>]` (default: `true`)<br/>Time shift features on the decoder side.
* `-keep_frequency [<boolean>]` (default: `false`)<br/>Keep frequency of words in dictionary.
* `-gsample <number>` (default: `0`)<br/>If not zero, extract a new sample from the corpus. In training mode, file sampling is done at each epoch. Values between 0 and 1 indicate ratio, values higher than 1 indicate data size
* `-gsample_dist <string>` (default: `''`)<br/>Configuration file with data class distribution to use for sampling training corpus. If not set, sampling is uniform.
* `-sort [<boolean>]` (default: `true`)<br/>If set, sort the sequences by size to build batches without source padding.
* `-shuffle [<boolean>]` (default: `true`)<br/>If set, shuffle the data (prior sorting).
* `-idx_files [<boolean>]` (default: `false`)<br/>If set, source and target files are 'key value' with key match between source and target.
* `-report_progress_every <number>` (default: `100000`)<br/>Report status every this many sentences.
* `-preprocess_pthreads <number>` (default: `4`)<br/>Number of parallel threads for preprocessing.

## Tokenizer options

* `-tok_src_mode <string>` (accepted: `conservative`, `aggressive`, `space`; default: `space`)<br/>Define how aggressive should the tokenization be. `space` is space-tokenization.
* `-tok_tgt_mode <string>` (accepted: ``conservative``, ``aggressive``, ``space``; default: `space`)<br/>Define how aggressive should the tokenization be. `space` is space-tokenization.
* `-tok_src_joiner_annotate [<boolean>]` (default: `false`)<br/>Include joiner annotation using `-joiner` character.
* `-tok_tgt_joiner_annotate [<boolean>]` (default: `false`)<br/>Include joiner annotation using `-joiner` character.
* `-tok_src_joiner <string>` (default: `￭`)<br/>Character used to annotate joiners.
* `-tok_tgt_joiner <string>` (default: `￭`)<br/>Character used to annotate joiners.
* `-tok_src_joiner_new [<boolean>]` (default: `false`)<br/>In `-joiner_annotate` mode, `-joiner` is an independent token.
* `-tok_tgt_joiner_new [<boolean>]` (default: `false`)<br/>In `-joiner_annotate` mode, `-joiner` is an independent token.
* `-tok_src_case_feature [<boolean>]` (default: `false`)<br/>Generate case feature.
* `-tok_tgt_case_feature [<boolean>]` (default: `false`)<br/>Generate case feature.
* `-tok_src_segment_case [<boolean>]` (default: `false`)<br/>Segment case feature, splits AbC to Ab C to be able to restore case
* `-tok_tgt_segment_case [<boolean>]` (default: `false`)<br/>Segment case feature, splits AbC to Ab C to be able to restore case
* `-tok_src_segment_alphabet <table>` (accepted: `Tagalog`, `Hanunoo`, `Limbu`, `Yi`, `Hebrew`, `Latin`, `Devanagari`, `Thaana`, `Lao`, `Sinhala`, `Georgian`, `Kannada`, `Cherokee`, `Kanbun`, `Buhid`, `Malayalam`, `Han`, `Thai`, `Katakana`, `Telugu`, `Greek`, `Myanmar`, `Armenian`, `Hangul`, `Cyrillic`, `Ethiopic`, `Tagbanwa`, `Gurmukhi`, `Ogham`, `Khmer`, `Arabic`, `Oriya`, `Hiragana`, `Mongolian`, `Kangxi`, `Syriac`, `Gujarati`, `Braille`, `Bengali`, `Tamil`, `Bopomofo`, `Tibetan`)<br/>Segment all letters from indicated alphabet.
* `-tok_tgt_segment_alphabet <table>` (accepted: ``Tagalog``, ``Hanunoo``, ``Limbu``, ``Yi``, ``Hebrew``, ``Latin``, ``Devanagari``, ``Thaana``, ``Lao``, ``Sinhala``, ``Georgian``, ``Kannada``, ``Cherokee``, ``Kanbun``, ``Buhid``, ``Malayalam``, ``Han``, ``Thai``, ``Katakana``, ``Telugu``, ``Greek``, ``Myanmar``, ``Armenian``, ``Hangul``, ``Cyrillic``, ``Ethiopic``, ``Tagbanwa``, ``Gurmukhi``, ``Ogham``, ``Khmer``, ``Arabic``, ``Oriya``, ``Hiragana``, ``Mongolian``, ``Kangxi``, ``Syriac``, ``Gujarati``, ``Braille``, ``Bengali``, ``Tamil``, ``Bopomofo``, ``Tibetan``)<br/>Segment all letters from indicated alphabet.
* `-tok_src_segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-tok_tgt_segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-tok_src_bpe_model <string>` (default: `''`)<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_tgt_bpe_model <string>` (default: `''`)<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_src_EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the end of token.
* `-tok_tgt_EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the end of token.
* `-tok_src_BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the beginning of token.
* `-tok_tgt_BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the beginning of token.
* `-tok_src_bpe_case_insensitive [<boolean>]` (default: `false`)<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_tgt_bpe_case_insensitive [<boolean>]` (default: `false`)<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_src_bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-tok_tgt_bpe_mode <string>` (accepted: ``suffix``, ``prefix``, ``both``, ``none``; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NOERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-seed <number>` (default: `3425`)<br/>Random seed.
