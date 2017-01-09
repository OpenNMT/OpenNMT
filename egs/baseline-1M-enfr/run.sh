#!/bin/bash
#
# Copyright 2017 Ubiqus (Author: Vincent Nguyen)
#		 Systran (Author Jean Senellart)
# MIT
#
# This recipe shows how to build an openNMT translation model from English to French
# based on a limited resource (1 Mio segments)
#
# Based on the tuto from the OpenNMT forum
# TODO replicate with the full Europarlv7 corpus

# TODO test is GPU is present or not
CUDA_VISIBLE_DEVICES=0


stage=0

# making these variables to make replication easier for other languages
sl=en
tl=fr

# Test set to be used for NIST evaluation
testset=newstest2014-$tl$sl
# testset=newsdiscusstest2015-$sl$tl

# At the moment only "stage" option is available anyway
. tools/parse_options.sh

# Data preparation
if [ $stage -le 0 ]; then
# TODO put this part in a local/download_data.sh script / test if Data is already downloaded / untarred
  mkdir -p data
  cd data
  wget https://s3.amazonaws.com/opennmt-trainingdata/baseline-1M-$sl$tl.tgz
  tar xzfv baseline-1M-$sl$tl.tgz --strip-components=1 -C .
  rm baseline-1M-$sl$tl.tgz
  cd ..
fi

# Tokenize the Corpus
# -mode: can be aggressive or conservative (default).
# conservative mode: letters, numbers and ‘_’ are kept in sequence, 
# hyphens are accepted as part of tokens. 
# Finally inner characters [.,] are also accepted (url, numbers).
# -sep_annotate: if set, add reversible separator mark to indicate separator-less or BPE tokenization
# (preference on symbol, then number, then letter)
# -case_feature: generate case feature - and convert all tokens to lowercase
#        N: not defined (for instance tokens without case)
#        L: token is lowercased (opennmt)
#        U: token is uppercased (OPENNMT)
#        C: token is capitalized (Opennmt)
#        M: token case is mixed (OpenNMT)
# -bpe_model: when set, activate BPE using the BPE model filename
if [ $stage -le 1 ]; then
#  for f in data/*.?? ; do th tools/tokenize.lua -case_feature -sep_annotate < $f > $f.tok ; done
  for f in data/*.?? ; do th tools/tokenize.lua < $f > $f.tok ; done
fi

# Preprocess the data
if [ $stage -le 2 ]; then
  mkdir -p exp
  th preprocess.lua -train_src data/generic-1M_train.$sl.tok \
  -train_tgt data/generic-1M_train.$tl.tok \
  -valid_src data/generic_valid.$sl.tok \
  -valid_tgt data/generic_valid.$tl.tok -save_data exp/model-$sl$tl
fi

# Train the model !!!! even if OS cuda device ID is 0 you need -gpuid=1
if [ $stage -le 3 ]; then
  th train.lua -data  exp/model-$sl$tl-train.t7 \
  -save_model exp/model-$sl$tl \
  -epochs 13 -learning_rate 1 -start_decay_at 5 -learning_rate_decay 0.65 -gpuid 1
  cp exp/model-$sl$tl"_epoch13_*.t7" exp/model-$sl$tl"_final.t7"
fi

# Deploy model for CPU usage
if [ $stage -le 4 ]; then
  th tools/release_model.lua -model exp/model-$sl$tl"_final.t7" -output_model exp/model-$sl$tl"_cpu.t7" -gpuid 1
fi

# Translate
if [ $stage -le 5 ]; then
  th translate.lua -model exp/model-$sl$tl"_cpu.t7" \
  -src data/generic_test.$sl.tok -output exp/generic_test.hyp.$tl.tok
fi

# Evaluate the generic test set with multi-bleu
if [ $stage -le 6 ]; then
  th tools/detokenize.lua < exp/generic_test.hyp.$tl.tok > exp/generic_test.hyp.$tl.detok
  tools/multi-bleu.perl data/generic_test.$tl \
  < exp/generic_test.hyp.$tl.detok > exp/generic_test_multibleu.txt
fi

###############################
#### Newstest Evaluation
####

if [ $stage -le 7 ]; then

  tools/input-from-sgm.perl < newstest/$testset-src.$sl.sgm \
  > newstest/$testset.$sl

  th tools/tokenize.lua < newstest/$testset.$sl \
  > newstest/$testset.$sl.tok

  th translate.lua -model exp/model-$sl$tl"_cpu".t7 \
  -src newstest/$testset.$sl.tok \
  -output exp/$testset.trans.$tl.tok


  th tools/tokenize.lua < exp/$testset.trans.$tl.tok \
  > exp/$testset.trans.$tl

# Wrap-xml to convert to sgm

  tools/wrap-xml.perl $tl newstest/$testset-src.$sl.sgm tst \
  < exp/$testset.trans.$tl \
  > exp/$testset.trans.$tl.sgm

  tools/mteval-v13a.pl -r newstest/$testset-ref.$tl.sgm \
  -s newstest/$testset-src.$sl.sgm -t exp/$testset.trans.$tl.sgm \
  -c > exp/nist-bleu-$testset
fi

