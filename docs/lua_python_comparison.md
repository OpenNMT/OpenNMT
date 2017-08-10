This page lists all the differences between commandline options of `OpenNMT` (lua version) and `OpenNMT-py` (python version). Identical options are not described.

# `tokenize.lua`, `learn_bpe.lua`

There is no python version yet of tokenization/bpe scripts.

# `preprocess.(lua|py)`

OpenNMT-lua reference option list for `preprocess.lua` is documented [here](http://opennmt.net/OpenNMT/options/preprocess/).

Options differences:

|      | `OpenNMT-lua` | `OpenNMT-py` |
| ---  | ---           | ---          |
| `-config` | | *not implemented* |
| `-save_config` | | *not supported* |
| `-src_type` | *not supported* | `text|img` |
| `-src_img_dir` | *not supported* | used for `-src_type img` |
| `-data_type` | `bitext|monotext|feattext` - used for ASR and LM models | |
| `-src_words_min_frequency` | keep all src vocab with this frequency | *not supported* |
| `-tgt_words_min_frequency` | keep all tgt vocab with this frequency | *not supported* |
| `-src_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `-tgt_seq_length_trunc` | *not supported* | if sentence too long, truncate it |
| `-check_plength` | check alignment of source/target for sequence tagging | *not supported* |
| `-time_shift_features` | shift feature by one timestep | *not supported* |
| `-keep_frequency` | used by sampled softmax | *not supported* |
| `-sort` | | *not optional*: can not be disabled |
| `-idx_files` | for `feattext` - provide format to align source&target alignment | *not supported* |
| `-lower` | *not supported* - is part of the tokenization features | runtime lowercasing |
