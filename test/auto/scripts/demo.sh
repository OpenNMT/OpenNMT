#!/bin/bash

th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data ${TMP}/demo ${PARAMS_PREPROCESS}
th train.lua -data ${TMP}/demo-train.t7 -save_model ${TMP}/model ${PARAMS_TRAIN}

