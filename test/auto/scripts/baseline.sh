#!/bin/bash

for i in ${DATA}/*.${SRC} ${DATA}/*.${TGT} ; do
  th tools/tokenize.lua ${PARAMS_TOKENIZE} < ${i} > ${i}.tok
  if [ $? -ne 0 ]; then
    err = $?
    echo "ERROR IN TOKENIZE - EXIT"
    exit err
  fi
done
if [ $? -ne 0 ]; then
  err = $?
  echo "ERROR IN TOKENIZE - EXIT"
  exit err
fi

th preprocess.lua -train_src ${DATA}/${NAME}_train.${SRC}.tok -train_tgt ${DATA}/${NAME}_train.${TGT}.tok -valid_src ${DATA}/${NAME}_valid.${SRC}.tok -valid_tgt ${DATA}/${NAME}_valid.${TGT}.tok -save_data ${DATA}/${NAME} ${PARAMS_PREPROCESS} > ${TMP}/preprocess.log
if [ $? -ne 0 ]; then
  err = $?
  echo "ERROR IN PREPROCESS - EXIT"
  exit err
fi

th train.lua -data ${TMP}/${NAME}-train.t7 -save_model ${TMP}/${NAME} ${PARAMS_TRAIN} > ${TMP}/train.log
if [ $? -ne 0 ]; then
  err = $?
  echo "ERROR IN TRAIN - EXIT"
  exit err
fi

# validations are checkpoints
grep "Validation perplexity:" ${TMP}/train.log | perl -pe 's/.*: /"CHECK __ VALPPL".(++$idx).": "/e'
