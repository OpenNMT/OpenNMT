#!/bin/bash

set -o xtrace

th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data ${TMP}/demo ${PARAMS_PREPROCESS} | tee ${TMP}/preprocess.log
if [ $? -ne 0 ]; then
  err = $?
  echo "-- ERROR IN PREPROCESS - EXIT"
  exit err
fi

# size of dictionaries are checkpoints
echo "-- CHECK __ SRCDICT:" `wc -l ${TMP}/demo.src.dict | perl -pe 's/ *(\d+) .*/$1/'`
echo "-- CHECK __ TGTDICT:" `wc -l ${TMP}/demo.tgt.dict | perl -pe 's/ *(\d+) .*/$1/'`

th train.lua -data ${TMP}/demo-train.t7 -save_model ${TMP}/model ${PARAMS_TRAIN} | tee ${TMP}/train.log
if [ $? -ne 0 ]; then
  err = $?
  echo "-- ERROR IN TRAIN - EXIT"
  exit err
fi

# validations are checkpoints
grep "Validation perplexity:" ${TMP}/train.log | perl -pe 's/.*: /"CHECK __ VALPPL".(++$idx).": "/e'
