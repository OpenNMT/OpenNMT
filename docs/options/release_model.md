<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

`release_model.lua` options:

* `-h`<br/>This help.
* `-md`<br/>Dump help in Markdown format.
* `-config <string>`<br/>Load options from this file.
* `-save_config <string>`<br/>Save options to this file.

## Model options

* `-model <string>`<br/>Path to the trained model to release.
* `-output_model <string>`<br/>Path the released model. If not set, the `release` suffix will be automatically added to the model filename.
* `-force`<br/>Force output model creation even if the target file exists.

## Cuda options

* `-gpuid <string>` (default: `0`)<br/>List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.
* `-fallback_to_cpu`<br/>If GPU can't be used, rollback on the CPU.
* `-fp16`<br/>Use half-precision float on GPU.
* `-no_nccl`<br/>Disable usage of nccl in parallel mode.

## Logger options

* `-log_file <string>`<br/>Output logs to a file under this path instead of stdout.
* `-disable_logs`<br/>If set, output nothing.
* `-log_level <string>` (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)<br/>Output logs at this level and above.

