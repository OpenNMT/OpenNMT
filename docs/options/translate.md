<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`translate.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Read options from config file.
* `-save_config <string>`<br/>Save options from config file.

## Data options

* `-src <string>`<br/>Source sequences to translate.
* `-tgt <string>`<br/>Optional true target sequences.
* `-output <string>`<br/>Output file. (default: `pred.txt`)

## Translator options

* `-model <string>`<br/>Path to the serialized model file.
* `-beam_size <number>`<br/>Beam size. (default: `5`)
* `-batch_size <number>`<br/>Batch size. (default: `30`)
* `-max_sent_length <number>`<br/>Maximum output sentence length. (default: `250`)
* `-replace_unk`<br/>Replace the generated <unk> tokens with the source token that has the highest attention weight. If phrase_table is provided, it will lookup the identified source token and give the corresponding target token. If it is not provided (or the identified source token does not exist in the table) then it will copy the source token
* `-phrase_table <string>`<br/>Path to source-target dictionary to replace <unk> tokens.
* `-n_best <number>`<br/>If > 1, it will also output an n_best list of decoded sentences. (default: `1`)
* `-max_num_unks <number>`<br/>All sequences with more <unk>s than this will be ignored during beam search. (default: `inf`)
* `-pre_filter_factor <number>`<br/>Optional, set this only if filter is being used. Before applying filters, hypotheses with top `beamSize * preFilterFactor` scores will be considered. If the returned hypotheses voilate filters, then set this to a larger value to consider more. (default: `1`)
* `-length_norm <number>`<br/>Length normalization coefficient (alpha). Hypotheses scores are divided by (5+|Y|/5 + 1)^alpha, where |Y| is current target length. If set to 0, no length normalization. (default: `0`)
* `-coverage_norm <number>`<br/>Coverage normalization coefficient (beta). An extra coverage term multiplied by beta is added to hypotheses scores. Coverage is expressed as a sum over all source words of a log of attention probabilities cumulated over target words. If is set to 0, no coverage normalization. (default: `0`)
* `-eos_norm <number>`<br/>End of sentence normalization coefficient (gamma). The score for the EOS token is multiplied by (|X|/|Y|)*gamma, where |X| is source length and |Y| is current target length. If set to 0, no EOS normalization. (default: `0`)

## Cuda options

* `-gpuid <string>`<br/>List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0. (default: `0`)
* `-fallback_to_cpu`<br/>If GPU can't be used, rollback on the CPU.
* `-fp16`<br/>Use half-precision float on GPU.
* `-no_nccl`<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>`<br/>Output logs at this level and above. (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)

## Other options

* `-time`<br/>Measure average translation time.

