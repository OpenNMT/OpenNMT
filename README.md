[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT) 


# OpenNMT: Open-Source Neural Machine Translation

<a href="https://opennmt.github.io/">OpenNMT</a> is a full-featured,
open-source (MIT) neural machine translation system utilizing the
[Torch](http://torch.ch) mathematical toolkit.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

The system is designed to be simple to use and easy to extend, while
maintaining efficiency and state-of-the-art translation
accuracy. Features include:

* Speed and memory optimizations for high-performance GPU training.
* Simple general-purpose interface, only requires and source/target data files.
* [C++ implementation of the translator](https://github.com/OpenNMT/CTranslate) for easy deployment.
* Extensions to allow other sequence generation tasks such as summarization and image captioning.

## Installation

OpenNMT only requires a Torch installation with few dependencies.

1. [Install Torch](http://torch.ch/docs/getting-started.html)
2. Install additional packages:

```bash
luarocks install tds
```

For other installation methods including Docker, visit the [documentation](http://opennmt.net/OpenNMT/installation/).

## Quickstart

OpenNMT consists of three commands:

1) Preprocess the data.

```
th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

2) Train the model.

```
th train.lua -data data/demo-train.t7 -save_model model
```

3) Translate sentences.

```
th translate.lua -model model_final.t7 -src data/src-test.txt -output pred.txt
```

For more details, visit the [documentation](http://opennmt.net/OpenNMT/).

## Citation

A <a href="https://arxiv.org/abs/1701.02810">technical report</a> on OpenNMT is available. If you use the system for academic work, please cite:

```
@ARTICLE{2017opennmt,
  author = {{Klein}, G. and {Kim}, Y. and {Deng}, Y. and {Senellart}, J. and {Rush}, A.~M.},
  title = "{OpenNMT: Open-Source Toolkit for Neural Machine Translation}",
  journal = {ArXiv e-prints},
  eprint = {1701.02810}
}
```

## Additional resources

* <a href="http://opennmt.github.io/OpenNMT">Documentation</a>
* <a href="http://opennmt.github.io/Models">Example models</a>
* <a href="http://forum.opennmt.net">Forum</a>
* <a href="https://gitter.im/OpenNMT/openmt">Gitter channel</a>
* <a href="https://demo-pnmt.systran.net">Live demo</a>
* <a href="http://opennmt.github.io/about">Bibliography</a>
