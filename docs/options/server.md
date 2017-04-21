<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`translation_server.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Load options from this file.
* `-save_config <string>`<br/>Save options to this file.

## Server options

* `-host <string>` (default: `127.0.0.1`)<br/>Host to run the server on.
* `-port <string>` (default: `5556`)<br/>Port to run the server on.

## Translator options

* `-model <string>`<br/>Path to the serialized model file.
* `-beam_size <number>` (default: `5`)<br/>Beam size.
* `-batch_size <number>` (default: `30`)<br/>Batch size.
* `-max_sent_length <number>` (default: `250`)<br/>Maximum output sentence length.
* `-replace_unk`<br/>Replace the generated <unk> tokens with the source token that has the highest attention weight. If `-phrase_table` is provided, it will lookup the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table) then it will copy the source token
* `-phrase_table <string>`<br/>Path to source-target dictionary to replace `<unk>` tokens.
* `-n_best <number>` (default: `1`)<br/>If > 1, it will also output an n-best list of decoded sentences.
* `-max_num_unks <number>` (default: `inf`)<br/>All sequences with more `<unk>`s than this will be ignored during beam search.
* `-pre_filter_factor <number>` (default: `1`)<br/>Optional, set this only if filter is being used. Before applying filters, hypotheses with top `beam_size * pre_filter_factor` scores will be considered. If the returned hypotheses voilate filters, then set this to a larger value to consider more.
* `-length_norm <number>` (default: `0`)<br/>Length normalization coefficient (alpha). If set to 0, no length normalization.
* `-coverage_norm <number>` (default: `0`)<br/>Coverage normalization coefficient (beta). An extra coverage term multiplied by beta is added to hypotheses scores. If is set to 0, no coverage normalization.
* `-eos_norm <number>` (default: `0`)<br/>End of sentence normalization coefficient (gamma). If set to 0, no EOS normalization.
* `-dump_input_encoding`<br/>Instead of generating target tokens conditional on the source tokens, we print the representation (encoding/embedding) of the input.

## Cuda options

* `-gpuid <string>` (default: `0`)<br/>List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu`<br/>If GPU can't be used, rollback on the CPU.
* `-fp16`<br/>Use half-precision float on GPU.
* `-no_nccl`<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

