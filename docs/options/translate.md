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
* `-batch_size <number>` (default: `30`)<br/>Batch size.
* `-idx_files [<boolean>]` (default: `false`)<br/>If set, source and target files are 'key value' with key match between source and target.

## Translator options

* `-model <string>` (default: `''`)<br/>Path to the serialized model file.
* `-beam_size <number>` (default: `5`)<br/>Beam size.
* `-max_sent_length <number>` (default: `250`)<br/>Maximum output sentence length.
* `-replace_unk [<boolean>]` (default: `false`)<br/>Replace the generated <unk> tokens with the source token that has the highest attention weight. If `-phrase_table` is provided, it will lookup the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table) then it will copy the source token
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

## Cuda options

* `-gpuid <table>` (default: `0`)<br/>List of GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu [<boolean>]` (default: `false`)<br/>If GPU can't be used, rollback on the CPU.
* `-fp16 [<boolean>]` (default: `false`)<br/>Use half-precision float on GPU.
* `-no_nccl [<boolean>]` (default: `false`)<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-time [<boolean>]` (default: `false`)<br/>Measure average translation time.
