[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT)

# OpenNMT: Open-Source Neural Machine Translation

<a href="https://opennmt.github.io/">OpenNMT</a> is a full-featured,
open-source (MIT) neural machine translation system utilizing the
[Torch](http://torch.ch) mathematical toolkit.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

The system is designed to be simple to use and easy to extend , while
maintaining efficiency and state-of-the-art translation
accuracy. Features include:

* Speed and memory optimizations for high-performance GPU training.
* Simple general-purpose interface, only requires and source/target data files.
* C-only decoder implementation for easy deployment.
* Extensions to allow other sequence generation tasks such as summarization and image captioning.

## Installation

OpenNMT only requires a vanilla torch/cutorch install. It uses `nn`, `nngraph`, and `cunn`.

Alternatively there is a (CUDA) Docker container available at <a href="https://hub.docker.com/r/harvardnlp/opennmt/">here</a>.


## Quickstart

OpenNMT consists of three commands:

1) Preprocess the data.

```th preprocess.lua -train_src_file data/src-train.txt -train_targ_file data/targ-train.txt -valid_src_file data/src-val.txt -valid_targ_file data/targ-val.txt -save_file data/demo```

2) Train the model.

```th train.lua -data data/demo-train.t7 -save_file model```

3) Translate sentences.

```th translate.lua -model model_epochX_PPL.t7 -src_file data/src-test.txt -output_file pred.txt```

See the <a href="http://opennmt.github.io/Guide">guide</a> for more details.

## Documentation

* <a href="http://opennmt.github.io/Guide">Options and Features</a> 
* <a href="http://opennmt.github.io/OpenNMT">Documentation</a> 
* <a href="http://opennmt.github.io/Models">Example Models</a>
* <a href="http://demo-pnmt.systran.net">Live Demo</a>
* <a href="http://opennmt.github.io/About">Bibliography</a>

