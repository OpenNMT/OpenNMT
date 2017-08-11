This page lists all the differences between commandline options of `OpenNMT` (lua version) and `OpenNMT-py` (python version). Identical options are not described.

# `tokenize.lua`, `learn_bpe.lua`

There is no python version yet of tokenization/bpe scripts.

# `preprocess.(lua|py)`

OpenNMT-lua reference option list for `preprocess.lua` is documented [here](http://opennmt.net/OpenNMT/options/preprocess/).

Options differences:

|      | `OpenNMT-lua` | `OpenNMT-py` |
| ---  | ---           | ---          |
| `config` | | *not implemented* |
| `save_config` | | *not supported* |
| `md` | dump help as md file for online documentation | *not supported* |
| `src_type` | *not supported* | `text`, `img` |
| `src_img_dir` | *not supported* | used for `src_type img` |
| `data_type` | `bitext`, `monotext`, `feattext` - used for ASR and LM models | |
| `src_words_min_frequency` | keep all src vocab with this frequency | *not supported* |
| `tgt_words_min_frequency` | keep all tgt vocab with this frequency | *not supported* |
| `src_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `tgt_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `check_plength` | check alignment of source/target for sequence tagging | *not supported* |
| `time_shift_features` | shift feature by one timestep | *not supported* |
| `keep_frequency` | used by sampled softmax | *not supported* |
| `sort` | | *not optional*: can not be disabled |
| `idx_files` | for `feattext` - provide format to align source&target alignment | *not supported* |
| `lower` | *not supported* - is part of the tokenization features | runtime lowercasing |
| `log_file`<br>`disable_logs`<br>`log_level` | | *not supported* |

# `train.(lua|py)`

OpenNMT-lua reference option list for `preprocess.lua` is documented [here](http://opennmt.net/OpenNMT/options/train/).

Options differences:

|      | `OpenNMT-lua` | `OpenNMT-py` |
| ---  | ---           | ---          |
| `config`<br>`save_config` |  | *not supported* |
| `sample`<br>`sample_type`<br>`sample_perplexity_init`<br>`sample_perplexity_max`<br>`sample_vocab`  | Sampled dataset options | *not supported* |
| `model_type` | `lm`, `seq2seq`, `seqtagger` | *not supported* |
| `enc_layers`<br>`dec_layers` | number of layers of the encoder/decoder | *not supported*: see `layers` |
| `src_word_vec_size`<br>`tgt_word_vec_size` |  | *not supported*: see `word_vec_size` and `feat_vec_size` |
| `fix_word_vecs_enc`<br>`fix_word_vecs_dec` | `true`, `false`, `pretrained` | *not supported* |
| `share_decoder_embeddings` | *not supported* | share the word and softmax embeddings for decoder |
| `feat_merge` | `concat`, `sum` | `concat`, `sum`, `mlp` |
| `dropout_input` | Dropout probability applied to the input of the recurrent module | *not supported* |
| `dropout_words` | Dropout probability applied to the source sequence | *not supported* |
| `dropout_type` | `naive`, `variational` | *not supported*: dropout is `naive` |
| `residual` | Add residual connections between recurrent layers | *not supported* |
| `bridge` | `copy`, `dense`, `dense_nonlinear` | *not supported* |
| `encoder_type` | `rnn`, `brnn`, `dbrnn`, `pdbrnn`, `gnmt`, `cnn` | `text`, `img`: if `text`, corresponding type is defined by `encoder_layer` and `brnn` |
| `encoder_layer` | *not supported* | `rnn`, `mean`, `transformer` |
| `decoder_layer` | *not supported* | `rnn`, `transformer` |
| `brnn` | defined by `encoder_type` | use a bidirectional encoder, if `encoder_layer` is `rnn` |
| `pdbrnn_reduction`<br>`pdbrnn_merge` | for `encoder_type` set to `pdbrnn` | *not supported* |
| `cnn_layers`<br>`cnn_kernel`<br>`cnn_size` | for `encoder_type` set to `cnn` | *not supported* |
| `truncated_decoder` | *not supported* | truncated back propagation through time |
| `attention` | `none`, `global` | *not supported*: only global attention |
| `attention_type` | see `global_attention` | `dot`, `general`, `mlp` |
| `global_attention` | `general`, `dot`, `concat` | see `attention_type` |
| `copy_attn` | *not supported* | copy attention layer |
| `coverage_attn`<br>`lambda_coverage` | *not supported* | coverage attention layer |
| `context_gate` | *not supported* | `source`, `target`, `both` |
| `use_pos_emb` | add positional embeddings to word embeddings | *not supported* |
| `max_pos` | connected to `use_pos_emb` | *not supported* |
| `position_encoding` | *not supported* | use a sinusoid to mark relative words positions. |
| `save_every` |  | *not supported* |
| `save_every_epochs` |  | *not supported* |
| `report_every` |  | *not supported* |
| `async_parallel`<br>`async_parallel_minbatch` | Async multi-gpu training | *not supported* |
| `start_iteration` |  | *not supported* |
| `end_epoch` | final epoch of the training | *not supported*: see `epochs` |
| `epochs` | *not supported*: see `end_epoch` | number of training epochs |
| `validation_metric` | `perplexity`, `loss`, `bleu`, `ter`, `dlratio` | *not supported* |
| `save_validation_translation_every` |  | *not supported* |
| `max_batch_size` | maximum batch size | *not supported*: see `max_generator_batches` |
| `uneven_batches` |  |  |
| `max_generator_batches` | *not supported* |  |
| `min_learning_rate` | do not continue the training past this learning rate value | *not supported* |
| `start_decay_score_delta` |  | *not supported* |
| `decay` | default, epoch_only, score_only | *not supported* |
| `decay_method` |  | use a custom learning rate decay (?) |
| `warmup_steps` | *not supported* | number of warmup steps for custom decay |
| `train_from_state_dict` | *not supported* |  |
| `continue` |  | *not supported* |
| `extra_shuffle` | *not supported* | shuffle and re-assign mini-batches |
| `start_checkpoint_at` | *not supported* |  |
| `gpus` | *not supported*: is `gpuid` | list of GPU identifiers for parallel training |
| `gpuid` | list of GPU identifiers for parallel training | is `gpus` |
| `log_interval` | *not supported* | print stats at this interval |
| `log_server` |  | *not supported*: is `exp_host` and `exp_port` |
| `exp_host, exp_port` | *not supported*: is `log_server` |  |
| `experiment_name` | *not supported*: is `exp` | crayon experiment name |
| `exp` | crayon experiment name | is `experiment name` |
| `fp16` | half-float precision for GPU | *not supported* |
| `fallback_to_cpu` |  | *not supported* |
| `no_nccl` |  | *not supported* |
