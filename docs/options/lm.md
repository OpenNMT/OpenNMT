<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`lm.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Data options

* `<mode>` (accepted: `score`, `sample`)<br/>'score' apply lm to input text, 'sample' samples output based on input text.
* `-src <string>` (required)<br/>Source sequences to sample/score.
* `-output <string>` (default: `output.txt`)<br/>Output file depend on `<mode>`.

## LM options

* `-model <string>` (required)<br/>Path to the serialized model file.
* `-batch_size <number>` (default: `30`)<br/>Batch size.
* `-max_length <number>` (default: `100`)<br/>Maximal length of sentences in sample mode.
* `-temperature <number>` (default: `1`)<br/>For `sample` mode, higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

## Cuda options

* `-gpuid <table>` (default: `0`)<br/>List of GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu [<boolean>]` (default: `false`)<br/>If GPU can't be used, rollback on the CPU.
* `-fp16 [<boolean>]` (default: `false`)<br/>Use half-precision float on GPU.
* `-no_nccl [<boolean>]` (default: `false`)<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout - if file name ending with json, output structure json.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NONE`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-time [<boolean>]` (default: `false`)<br/>Measure average translation time.
