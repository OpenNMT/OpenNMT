#!/bin/bash

set -x

echo "default tokenization"
for f in test/data/*.txt ; do th tools/tokenize.lua -joiner_annotate < ${f} > ${f%.txt}_1.tok ; done

echo "parallel tokenization with joiner annotation and case"
for f in test/data/*.txt ; do th tools/tokenize.lua -joiner_annotate < ${f} > ${f%.txt}_2.tok -case_feature -nparallel 2 ; done

echo "default preprocess 1"
th preprocess.lua -train_src test/data/train.en_1.tok -train_tgt test/data/train.fr_1.tok \
            -valid_src test/data/valid.en_1.tok -valid_tgt test/data/valid.fr_1.tok -save_data test/data/train_1

echo "default preprocess 2"
th preprocess.lua -train_src test/data/train.en_2.tok -train_tgt test/data/train.fr_2.tok \
            -valid_src test/data/valid.en_2.tok -valid_tgt test/data/valid.fr_2.tok -save_data test/data/train_2


echo "baseline training - CPU"
th train.lua -data test/data/train_2-train.t7 -save_model model2a -end_epoch 1 -rnn_size 50 -word_vec_size 20 -layers 1 -input_feed 0 -profiler

echo "baseline training - GPU"
th train.lua -data test/data/train_2-train.t7 -save_model model2b -end_epoch 2 -profiler -gpuid 1

echo "baseline training - GPU parallel"
th train.lua -data test/data/train_2-train.t7 -save_model model2c -end_epoch 2 -profiler -max_batch_size 32 -gpuid 1,2

echo "bigger model - GPU"
th train.lua -data test/data/train_2-train.t7 -save_model model2d -end_epoch 2 -profiler -gpuid 1 -layers 4 -rnn_size 800 -word_vec_size 800

echo "model with residual & brnn merge - GPU"
th train.lua -data test/data/train_2-train.t7 -save_model model2e -end_epoch 2 -profiler -gpuid 1 -brnn -residual

echo "model with brnn  - GPU"
th train.lua -data test/data/train_2-train.t7 -save_model model2f -end_epoch 2 -profiler -gpuid 1 -brnn -brnn_merge concat

