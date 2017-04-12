#!/bin/bash

th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data ${TMP}/demo ${PARAMS_PREPROCESS} > ${TMP}/preprocess.log

# size of dictionaries are checkpoints
echo "CHECK __ SRCDICT:" `wc -l ${TMP}/demo.src.dict | perl -pe 's/ *(\d+) .*/$1/'`
echo "CHECK __ TGTDICT:" `wc -l ${TMP}/demo.tgt.dict | perl -pe 's/ *(\d+) .*/$1/'`

th train.lua -data ${TMP}/demo-train.t7 -save_model ${TMP}/model ${PARAMS_TRAIN} > ${TMP}/train.log

# validations are checkpoints
grep "Validation perplexity:" ${TMP}/train.log | perl -pe 's/.*: /"CHECK __ VALPPL".(++$idx).": "/e'
