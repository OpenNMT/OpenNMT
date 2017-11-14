<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`translate.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Data options

* `-src <string>` (required)<br/>Source sequences to translate.
* `-tgt <string>` (default: `''`)<br/>Optional true target sequences.
* `-output <string>` (default: `pred.txt`)<br/>Output file.
* `-save_attention <string>` (default: `''`)<br/>Optional attention output file.
* `-batch_size <number>` (default: `30`)<br/>Batch size.
* `-idx_files [<boolean>]` (default: `false`)<br/>If set, source and target files are 'key value' with key match between source and target.

## Translator options

* `-model <string>` (default: `''`)<br/>Path to the serialized model file.
* `-lm_model <string>` (default: `''`)<br/>Path to serialized language model file.
* `-lm_weight <number>` (default: `0.1`)<br/>Relative weight of language model.
* `-beam_size <number>` (default: `5`)<br/>Beam size.
* `-max_sent_length <number>` (default: `250`)<br/>Maximum output sentence length.
* `-replace_unk [<boolean>]` (default: `false`)<br/>Replace the generated <unk> tokens with the source token that has the highest attention weight. If `-phrase_table` is provided, it will lookup the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table) then it will copy the source token
* `-replace_unk_tagged [<boolean>]` (default: `false`)<br/>The same as -replace_unk, but wrap the replaced token in ｟unk:xxxxx｠ if it is not found in the phrase table.
* `-lexical_constraints [<boolean>]` (default: `false`)<br/>Force the beam search to apply the translations from the phrase table.
* `-limit_lexical_constraints [<boolean>]` (default: `false`)<br/>Prevents producing each lexical constraint more than required.
* `-phrase_table <string>` (default: `''`)<br/>Path to source-target dictionary to replace `<unk>` tokens.
* `-n_best <number>` (default: `1`)<br/>If > 1, it will also output an n-best list of decoded sentences.
* `-max_num_unks <number>` (default: `inf`)<br/>All sequences with more `<unk>`s than this will be ignored during beam search.
* `-target_subdict <string>` (default: `''`)<br/>Path to target words dictionary corresponding to the source.
* `-pre_filter_factor <number>` (default: `1`)<br/>Optional, set this only if filter is being used. Before applying filters, hypotheses with top `beam_size * pre_filter_factor` scores will be considered. If the returned hypotheses voilate filters, then set this to a larger value to consider more.
* `-length_norm <number>` (default: `0`)<br/>Length normalization coefficient (alpha). If set to 0, no length normalization.
* `-coverage_norm <number>` (default: `0`)<br/>Coverage normalization coefficient (beta). An extra coverage term multiplied by beta is added to hypotheses scores. If is set to 0, no coverage normalization.
* `-eos_norm <number>` (default: `0`)<br/>End of sentence normalization coefficient (gamma). If set to 0, no EOS normalization.
* `-dump_input_encoding [<boolean>]` (default: `false`)<br/>Instead of generating target tokens conditional on the source tokens, we print the representation (encoding/embedding) of the input.
* `-save_beam_to <string>` (default: `''`)<br/>Path to a file where the beam search exploration will be saved in a JSON format. Requires the `dkjson` package.

## Tokenizer options

* `-tok_src_mode <string>` (accepted: `conservative`, `aggressive`, `space`; default: `space`)<br/>Define how aggressive should the tokenization be. `space` is space-tokenization.
* `-tok_tgt_mode <string>` (accepted: `conservative`, `aggressive`, `space`; default: `space`)<br/>Define how aggressive should the tokenization be. `space` is space-tokenization.
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
* `-tok_tgt_segment_alphabet <table>` (accepted: `Tagalog`, `Hanunoo`, `Limbu`, `Yi`, `Hebrew`, `Latin`, `Devanagari`, `Thaana`, `Lao`, `Sinhala`, `Georgian`, `Kannada`, `Cherokee`, `Kanbun`, `Buhid`, `Malayalam`, `Han`, `Thai`, `Katakana`, `Telugu`, `Greek`, `Myanmar`, `Armenian`, `Hangul`, `Cyrillic`, `Ethiopic`, `Tagbanwa`, `Gurmukhi`, `Ogham`, `Khmer`, `Arabic`, `Oriya`, `Hiragana`, `Mongolian`, `Kangxi`, `Syriac`, `Gujarati`, `Braille`, `Bengali`, `Tamil`, `Bopomofo`, `Tibetan`)<br/>Segment all letters from indicated alphabet.
* `-tok_src_segment_numbers [<boolean>]` (default: `false`)<br/>Segment numbers into single digits.
* `-tok_tgt_segment_numbers [<boolean>]` (default: `false`)<br/>Segment numbers into single digits.
* `-tok_src_segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-tok_tgt_segment_alphabet_change [<boolean>]` (default: `false`)<br/>Segment if alphabet change between 2 letters.
* `-tok_src_bpe_model <string>` (default: `''`)<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_tgt_bpe_model <string>` (default: `''`)<br/>Apply Byte Pair Encoding if the BPE model path is given. If the option is used, BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_src_bpe_EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the End of Token while applying BPE in mode 'prefix' or 'both'.
* `-tok_tgt_bpe_EOT_marker <string>` (default: `</w>`)<br/>Marker used to mark the End of Token while applying BPE in mode 'prefix' or 'both'.
* `-tok_src_bpe_BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the Beginning of Token while applying BPE in mode 'suffix' or 'both'.
* `-tok_tgt_bpe_BOT_marker <string>` (default: `<w>`)<br/>Marker used to mark the Beginning of Token while applying BPE in mode 'suffix' or 'both'.
* `-tok_src_bpe_case_insensitive [<boolean>]` (default: `false`)<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_tgt_bpe_case_insensitive [<boolean>]` (default: `false`)<br/>Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`.
* `-tok_src_bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-bpe_BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-bpe_EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-tok_tgt_bpe_mode <string>` (accepted: `suffix`, `prefix`, `both`, `none`; default: `suffix`)<br/>Define the BPE mode. This option will be overridden/set automatically if the BPE model specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-bpe_BOT_marker` to the begining of each word to learn prefix-oriented pair statistics; `suffix`: append `-bpe_EOT_marker` to the end of each word to learn suffix-oriented pair statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`: no `suffix` nor `prefix`.
* `-tok_src_normalize_cmd <string>` (default: `''`)<br/>Command for on-the-fly corpus normalization. It should work in 'pipeline' mode.
* `-tok_tgt_normalize_cmd <string>` (default: `''`)<br/>Command for on-the-fly corpus normalization. It should work in 'pipeline' mode.
* `-detokenize_output [<boolean>]` (default: `false`)<br/>Detokenize output.

## Cuda options

* `-gpuid <table>` (default: `0`)<br/>List of GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu [<boolean>]` (default: `false`)<br/>If GPU can't be used, rollback on the CPU.
* `-fp16 [<boolean>]` (default: `false`)<br/>Use half-precision float on GPU.
* `-no_nccl [<boolean>]` (default: `false`)<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout - if file name ending with json, output structure json.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NOERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-time [<boolean>]` (default: `false`)<br/>Measure average translation time.
