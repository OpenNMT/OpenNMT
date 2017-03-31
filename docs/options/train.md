# train.lua

* `-h`: This help.
* `-md`: Dump help in Markdown format.
* `-config`: Read options from config file.
* `-save_config`: Save options from config file.

## Data options

* `-data`: Path to the data package *-train.t7 file from preprocess.lua.
* `-save_model`: Model filename (the model will be saved as <save_model>_epochN_PPL.t7 where PPL is the validation perplexity.

## Sampled dataset options

* `-sample`: Number of instances to sample from train data in each epoch. (default: `0`)
* `-sample_w_ppl`: If set, ese perplexity as a probability distribution when sampling.
* `-sample_w_ppl_init`: Start perplexity-based sampling when average train perplexity per batch falls below this value. (default: `15`)
* `-sample_w_ppl_max`: When greater than 0, instances with perplexity above this value will be considered as noise and ignored; when less than 0, mode + (-sample_w_ppl_max) * stdev will be used as threshold. (default: `-1.5`)

## Model options

* `-model_type`: Type of model to train. This option impacts all options choices. (accepted: `lm`, `seq2seq`, `seqtagger`; default: `seq2seq`)
* `-param_init`: Parameters are initialized over uniform distribution with support (-param_init, param_init). (default: `0.1`)

## Sequence to Sequence with Attention options

* `-word_vec_size`: Shared word embedding size. If set, this overrides src_word_vec_size and tgt_word_vec_size. (default: `0`)
* `-src_word_vec_size`: Comma-separated list of source embedding sizes: word[,feat1,feat2,...]. (default: `500`)
* `-tgt_word_vec_size`: Comma-separated list of target embedding sizes: word[,feat1,feat2,...]. (default: `500`)
* `-pre_word_vecs_enc`: Path to pretrained word embeddings on the encoder side serialized as a Torch tensor.
* `-pre_word_vecs_dec`: Path to pretrained word embeddings on the decoder side serialized as a Torch tensor.
* `-fix_word_vecs_enc`: Fix word embeddings on the encoder side. (accepted: `0`, `1`; default: `0`)
* `-fix_word_vecs_dec`: Fix word embeddings on the decoder side. (accepted: `0`, `1`; default: `0`)
* `-feat_merge`: Merge action for the features embeddings. (accepted: `concat`, `sum`; default: `concat`)
* `-feat_vec_exponent`: When features embedding sizes are not set and using -feat_merge concat, their dimension will be set to N^exponent where N is the number of values the feature takes. (default: `0.7`)
* `-feat_vec_size`: When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features (default: `20`)
* `-input_feed`: Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder. (accepted: `0`, `1`; default: `1`)
* `-layers`: Number of recurrent layers of the encoder and decoder. (default: `2`)
* `-rnn_size`: Hidden size of the recurrent unit. (default: `500`)
* `-rnn_type`: Type of recurrent cell. (accepted: `LSTM`, `GRU`; default: `LSTM`)
* `-dropout`: Dropout probability applied between recurrent layers. (default: `0.3`)
* `-dropout_input`: Also apply dropout to the input of the recurrent module.
* `-residual`: Add residual connections between recurrent layers.
* `-brnn`: Use a bidirectional encoder.
* `-dbrnn`: Use a deep bidirectional encoder.
* `-pdbrnn`: Use a pyramidal deep bidirectional encoder.
* `-brnn_merge`: Merge action for the bidirectional states. (accepted: `concat`, `sum`; default: `sum`)
* `-pdbrnn_reduction`: Time-reduction factor at each layer. (default: `2`)

## Optimization options

* `-max_batch_size`: Maximum batch size. (default: `64`)
* `-uneven_batches`: If true, batches are filled up to max_batch_size even if source lengths are different. Slower but needed for some tasks.
* `-optim`: Optimization method. (accepted: `sgd`, `adagrad`, `adadelta`, `adam`; default: `sgd`)
* `-learning_rate`: Starting learning rate. If adagrad or adam is used, then this is the global learning rate. Recommended settings are: sgd = 1, adagrad = 0.1, adam = 0.0002. (default: `1`)
* `-min_learning_rate`: Do not continue the training past this learning rate value. (default: `0`)
* `-max_grad_norm`: Clip the gradients norm to this value. (default: `5`)
* `-learning_rate_decay`: Learning rate decay factor: learning_rate = learning_rate * learning_rate_decay. (default: `0.7`)
* `-start_decay_at`: In "default" decay mode, start decay after this epoch. (default: `9`)
* `-start_decay_ppl_delta`: Start decay when validation perplexity improvement is lower than this value. (default: `0`)
* `-decay`: When to apply learning rate decay. "default": decay after each epoch past start_decay_at or as soon as the validation perplexity is not improving more than start_decay_ppl_delta, "perplexity_only": only decay when validation perplexity is not improving more than start_decay_ppl_delta. (accepted: `default`, `perplexity_only`; default: `default`)

## Trainer options

* `-save_every`: Save intermediate models every this many iterations within an epoch. If = 0, will not save intermediate models. (default: `5000`)
* `-report_every`: Report progress every this many iterations within an epoch. (default: `50`)
* `-async_parallel`: When training on multiple GPUs, update parameters asynchronously.
* `-async_parallel_minbatch`: In asynchronous training, minimal number of sequential batches before being parallel. (default: `1000`)
* `-start_iteration`: If loading from a checkpoint, the iteration from which to start. (default: `1`)
* `-start_epoch`: If loading from a checkpoint, the epoch from which to start. (default: `1`)
* `-end_epoch`: The final epoch of the training. (default: `13`)
* `-curriculum`: For this many epochs, order the minibatches based on source length. Sometimes setting this to 1 will increase convergence speed. (default: `0`)

## Checkpoint options

* `-train_from`: Path to a checkpoint.
* `-continue`: If set, continue the training where it left off.

## Crayon options

* `-exp_host`: Crayon server IP. (default: `127.0.0.1`)
* `-exp_port`: Crayon server port. (default: `8889`)
* `-exp`: Crayon experiment name.

## Cuda options

* `-gpuid`: List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0. (default: `0`)
* `-fallback_to_cpu`: If GPU can't be used, rollback on the CPU.
* `-fp16`: Use half-precision float on GPU.
* `-no_nccl`: Disable usage of nccl in parallel mode.

## Logger options

* `-log_file`: Output logs to a file under this path instead of stdout.
* `-disable_logs`: If set, output nothing.
* `-log_level`: Output logs at this level and above. (accepted: `DEBUG`, `INFO`, `WARNING`, `ERROR`; default: `INFO`)

## Other options

* `-disable_mem_optimization`: Disable sharing of internal buffers between clones for visualization or development.
* `-profiler`: Generate profiling logs.
* `-seed`: Random seed. (default: `3435`)

