<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`tag.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Read options from config file.
* `-save_config <string>`<br/>Save options from config file.

## Data options

* `-src <string>`<br/>Source sequences to tag.
* `-output <string>` (default: `pred.txt`)<br/>Output file.

## Tagger options

* `-model <string>`<br/>Path to the serialized model file.
* `-batch_size <number>` (default: `30`)<br/>Batch size.

## Cuda options

* `-gpuid <string>` (default: `0`)<br/>List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu`<br/>If GPU can't be used, rollback on the CPU.
* `-fp16`<br/>Use half-precision float on GPU.
* `-no_nccl`<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

## Other options

* `-time`<br/>Measure average translation time.

