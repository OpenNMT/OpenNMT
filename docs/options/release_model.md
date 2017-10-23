<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`release_model.lua` options:

* `-h [<boolean>]` (default: `false`)<br/>This help.
* `-md [<boolean>]` (default: `false`)<br/>Dump help in Markdown format.
* `-config <string>` (default: `''`)<br/>Load options from this file.
* `-save_config <string>` (default: `''`)<br/>Save options to this file.

## Model options

* `-model <string>` (required)<br/>Path to the trained model to release.
* `-output_model <string>` (default: `''`)<br/>Path the released model. If not set, the `release` suffix will be automatically added to the model filename.
* `-force [<boolean>]` (default: `false`)<br/>Force output model creation even if the target file exists.

## Cuda options

* `-gpuid <table>` (default: `0`)<br/>List of GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu [<boolean>]` (default: `false`)<br/>If GPU can't be used, rollback on the CPU.
* `-fp16 [<boolean>]` (default: `false`)<br/>Use half-precision float on GPU.
* `-no_nccl [<boolean>]` (default: `false`)<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>` (default: `''`)<br/>Output logs to a file under this path instead of stdout - if file name ending with json, output structure json.
* `-disable_logs [<boolean>]` (default: `false`)<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `NOERROR`; default: `INFO`)<br/>Output logs at this level and above.
