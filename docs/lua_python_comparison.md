This page lists all the differences between commandline options of `OpenNMT` (lua version) and `OpenNMT-py` (python version). Identical options are not described.

## `tokenize.lua`, `learn_bpe.lua`

There is no python version yet of tokenization/bpe scripts.

## `preprocess.(lua|py)`

OpenNMT-lua reference option list for `preprocess.lua` is documented [here](http://opennmt.net/OpenNMT/options/preprocess/).

Commandline options differences:

|      | `OpenNMT-lua` | `OpenNMT-py` |
| ---  | ---           | ---          |
| `config`<br>`save_config` |  | *not supported* |
| `src_img_dir` | *not supported* | used for `src_type img` |
| `data_type` | `bitext`, `monotext`, `feattext` - used for ASR and LM models | `text`, `img` (*) |
| `src_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `tgt_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `check_plength` | check alignment of source/target for sequence tagging | *not supported* |
| `time_shift_features` | shift feature by one timestep | *not supported* |
| `keep_frequency` | used by sampled softmax | *not supported* |
| `sort` | | *not optional*: can not be disabled |
| `idx_files` | for `feattext` - provide format to align source&target alignment | *not supported* |
| <span style="color:blue">Logging Options<span> |||
| `log_file`<br>`disable_logs`<br>`log_level` | | *not supported* |

(*) Lua implementation of `im2text` is in independent repository [here](https://github.com/OpenNMT/Im2Text).

## `train.(lua|py)`

OpenNMT-lua reference option list for `preprocess.lua` is documented [here](http://opennmt.net/OpenNMT/options/train/).

Commandline options differences:

|      | `OpenNMT-lua` | `OpenNMT-py` |
| ---  | ---           | ---          |
| `config`<br>`save_config` |  | *not supported* |
| <span style="color:blue">Sampling Options<span> |||
| `sample`<br>`sample_type`<br>`sample_perplexity_init`<br>`sample_perplexity_max`<br>`sample_vocab`  | [Sampled dataset options](/options/train/#sampled-dataset-options) | *not supported* |
| <span style="color:blue">Model Options<span> |||
| `model_type` | `lm`, `seq2seq`, `seqtagger` | `text`, `img` |
| `share_decoder_embeddings` | *not supported* | share the word and softmax embeddings for decoder |
| `use_pos_emb` | add positional embeddings to word embeddings | *not supported* |
| `max_pos` | connected to `use_pos_emb` | *not supported* |
| `position_encoding` | *not supported* | use a sinusoid to mark relative words positions. |
| `feat_merge` | `concat`, `sum` | `concat`, `sum`, `mlp` |
| `dropout_input` | Dropout applied to the input of the recurrent module | *not supported* |
| `dropout_words` | Dropout applied to the full source sequence | *not supported* |
| `dropout_type` | `naive`, `variational` | *not supported*: dropout is `naive` |
| `residual` | Add residual connections between recurrent layers | *not supported* |
| `bridge` | `copy`, `dense`, `dense_nonlinear` | *not supported* |
| `encoder_type` | `rnn`, `brnn`, `dbrnn`, `pdbrnn`, `gnmt`, `cnn` | `rnn`, `mean`, `transformer` |
| `decoder_layer` | *not supported* | `rnn`, `transformer` |
| `pdbrnn_reduction`<br>`pdbrnn_merge` | for `encoder_type` set to `pdbrnn` | *not supported* |
| `cnn_layers`<br>`cnn_kernel`<br>`cnn_size` | for `encoder_type` set to `cnn` | *not supported* |
| `truncated_decoder` | *not supported* | truncated back propagation through time |
| <span style="color:blue">Attention Options<span> |||
| `attention` | `none`, `global` | *not supported*: only global attention |
| `global_attention` | `general`, `dot`, `concat` | `dot`, `general`, `mlp` |
| `copy_attn` | *not supported* | copy attention layer |
| `coverage_attn`<br>`lambda_coverage` | *not supported* | coverage attention layer |
| `context_gate` | *not supported* | `source`, `target`, `both` |
| <span style="color:blue">Training Options<span> |||
| `async_parallel`<br>`async_parallel_minbatch` | Async multi-gpu training | *not supported* |
| `start_iteration` |  | *not supported* |
| `end_epoch` | final epoch of the training | *not supported*: see `epochs` |
| `epochs` | *not supported*: see `end_epoch` | number of training epochs |
| `validation_metric` | `perplexity`, `loss`, `bleu`, `ter`, `dlratio` | *not supported* always perplexity |
| `save_validation_translation_every` |  | *not supported* |
| <span style="color:blue">Optim Options<span> |||
| `max_batch_size` | maximum batch size | *not supported*: see `max_generator_batches` |
| `uneven_batches` |  |  |
| `max_generator_batches` | *not supported* |  |
| `min_learning_rate` | do not continue the training past this learning rate value | *not supported* |
| `start_decay_score_delta` |  | *not supported* |
| `decay` | default, epoch_only, score_only | *not supported* |
| `decay_method` |  | use a custom learning rate decay (?) |
| `warmup_steps` | *not supported* | number of warmup steps for custom decay |
| <span style="color:blue">Saver Options<span> |||
| `continue` |  | *not supported* |
| `start_checkpoint_at` | *not supported* |  |
| `save_every` |  | *not supported* |
| `save_every_epochs` |  | *not supported* |
| <span style="color:blue">Logging Options<span> |||
| `log_file`<br>`disable_logs`<br>`log_level` | | *not supported* |
| `exp_port` | port of the Crayon server | *not supported*: default port 8889 is used |
| <span style="color:blue">GPU Options<span> |||
| `fp16` | half-float precision for GPU | *not supported* |
| `fallback_to_cpu` |  | *not supported* |
| `no_nccl` |  | *not supported* |
