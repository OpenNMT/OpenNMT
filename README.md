## Sequence-to-Sequence Learning with Attentional Neural Networks

[Torch](http://torch.ch) implementation of a standard sequence-to-sequence model with (optional)
attention where the encoder-decoder are LSTMs. Encoder can be a bidirectional LSTM.
Additionally has the option to use characters
(instead of input word embeddings) by running a convolutional neural network followed by a
[highway network](http://arxiv.org/abs/1505.00387) over character embeddings to use as inputs.

The attention model is from
[Effective Approaches to Attention-based
Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf),
Luong et al. EMNLP 2015. We use the *global-general-attention* model with the
*input-feeding* approach from the paper. Input-feeding is optional and can be turned off.

The character model is from [Character-Aware Neural
Language Models](http://arxiv.org/abs/1508.06615), Kim et al. AAAI 2016.

There are a lot of additional options on top of the baseline model, mainly thanks to the fantastic folks 
at [SYSTRAN](http://www.systransoft.com). Specifically, there are functionalities which implement:
* [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf). Luong et al., EMNLP 2015.
* [Character-based Neural Machine Translation](https://aclweb.org/anthology/P/P16/P16-2058.pdf). Costa-Jussa and Fonollosa, ACL 2016.
* [Compression of Neural Machine Translation Models via Pruning](https://arxiv.org/pdf/1606.09274.pdf). See et al., COLING 2016.
* [Sequence-Level Knowledge Distillation](https://arxiv.org/pdf/1606.07947.pdf). Kim and Rush., EMNLP 2016.
* [Deep Recurrent Models with Fast Forward Connections for Neural Machine Translation](https://arxiv.org/pdf/1606.04199).
Zhou et al, TACL 2016.
* [Guided Alignment Training for Topic-Aware Neural Machine Translation](https://arxiv.org/pdf/1607.01628). Chen et al., arXiv:1607.01628.
* [Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/pdf/1606.02892). Senrich et al., arXiv:1606.02892

See below for more details on how to use them.

This project is maintained by [Yoon Kim](http://people.fas.harvard.edu/~yoonkim).
Feel free to post any questions/issues on the issues page.

### Dependencies

#### Lua
You will need the following packages:
* nn
* nngraph

GPU usage will additionally require:
* cutorch
* cunn

If running the character model, you should also install:
* cudnn
* luautf8




#### Using additional input features
[Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/abs/1606.02892) (Senrich et al. 2016) shows that translation performance can be increased by using additional input features.

Similarly to this work, you can annotate each word in the **source** text by using the `-|-` separator:

```
word1-|-feat1-|-feat2 word2-|-feat1-|-feat2
```

It supports an arbitrary number of features with arbitrary labels. However, all input words must have the **same** number of annotations. See for example `data/src-train-case.txt` which annotates each word with the case information.

To evaluate the model, the option `-feature_dict_prefix` is required on `evaluate.lua` which points to the prefix of the features dictionnaries generated during the preprocessing.

#### Pruning a model

[Compression of Neural Machine Translation Models via Pruning](http://arxiv.org/pdf/1606.09274v1.pdf) (See et al. 2016) shows that a model can be aggressively pruned while keeping the same performace.

To prune a model - you can use `prune.lua` which implement class-bind, and class-uniform pruning technique from the paper.

* `model`: the model to prune
* `savefile`: name of the pruned model
* `gpuid`: Which gpu to use. -1 = use CPU. Depends if the model is serialized for GPU or CPU
* `ratio`: pruning rate
* `prune`: pruning technique `blind` or `uniform`, by default `blind`

note that the pruning cut connection with lowest weight in the linear models by using a boolean mask. The size of the file is a little larger since it stores the actual full matrix and the binary mask.

Models can be retrained - typically you can recover full capacity of a model pruned at 60% or even 80% by few epochs of additional trainings.

#### Switching between GPU/CPU models
By default, the model will always save the final model as a CPU model, but it will save the
intermediate models as a CPU/GPU model depending on how you specified `-gpuid`.
If you want to run beam search on the CPU with an intermediate model trained on the GPU,
you can use `convert_to_cpu.lua` to convert the model to CPU and run beam search.

#### GPU memory requirements/Training speed
Training large sequence-to-sequence models can be memory-intensive. Memory requirements will
dependent on batch size, maximum sequence length, vocabulary size, and (obviously) model size.
Here are some benchmark numbers on a GeForce GTX Titan X.
(assuming batch size of 64, maximum sequence length of 50 on both the source/target sequence,
vocabulary size of 50000, and word embedding size equal to rnn size):

(`prealloc = 0`)
* 1-layer, 100 hidden units: 0.7G, 21.5K tokens/sec
* 1-layer, 250 hidden units: 1.4G, 14.1K tokens/sec
* 1-layer, 500 hidden units: 2.6G, 9.4K tokens/sec
* 2-layers, 500 hidden units: 3.2G, 7.4K tokens/sec
* 4-layers, 1000 hidden units: 9.4G, 2.5K tokens/sec

Thanks to some fantastic work from folks at [SYSTRAN](http://www.systransoft.com), turning `prealloc` on
will lead to much more memory efficient training

(`prealloc = 1`)
* 1-layer, 100 hidden units: 0.5G, 22.4K tokens/sec
* 1-layer, 250 hidden units: 1.1G, 14.5K tokens/sec
* 1-layer, 500 hidden units: 2.1G, 10.0K tokens/sec
* 2-layers, 500 hidden units: 2.3G, 8.2K tokens/sec
* 4-layers, 1000 hidden units: 6.4G, 3.3K tokens/sec

Tokens/sec refers to total (i.e. source + target) tokens processed per second.
If using different batch sizes/sequence length, you should (linearly) scale
the above numbers accordingly. You can make use of memory on multiple GPUs by using
`-gpuid2` option in `train.lua`. This will put the encoder on the GPU specified by
`-gpuid`, and the decoder on the GPU specified by `-gpuid2`.

#### Evaluation
For translation, evaluation via BLEU can be done by taking the output from `beam.lua` and using the
`multi-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder). For example

```
perl multi-bleu.perl gold.txt < pred.txt
```

#### Evaluation of States and Attention
attention_extraction.lua can be used to extract the attention and the LSTM states. It uses the following (required) options:

* `model`: Path to model .t7 file.
* `src_file`: Source sequence to decode (one line per sequence).
* `targ_file`: True target sequence.
* `src_dict`: Path to source vocabulary (`*.src.dict` file from `preprocess.py`).
* `targ_dict`: Path to target vocabulary (`*.targ.dict` file from `preprocess.py`).

Output of the script are two files, `encoder.hdf5` and `decoder.hdf5`. The encoder contains the states for every layer of the encoder LSTM and the offsets for the start of each source sentence. The decoder contains the states for the decoder LSTM layers and the offsets for the start of gold sentence. It additionally contains the attention for each time step (if the model uses attention).


#### Pre-trained models
We've uploaded English <-> German models trained on 4 million sentences from
[Workshop on Machine Translation 2015](http://www.statmt.org/wmt15/translation-task.html).
Download link is below:

https://drive.google.com/open?id=0BzhmYioWLRn_aEVnd0ZNcWd0Y2c

These models are 4-layer LSTMs with 1000 hidden units and essentially replicates the results from
[Effective Approaches to Attention-based
Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf),
Luong et al. EMNLP 2015.

#### Acknowledgments
Our implementation utilizes code from the following:
* [Andrej Karpathy's char-rnn repo](https://github.com/karpathy/char-rnn)
* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)
* [Element rnn library](https://github.com/Element-Research/rnn)

#### Licence
MIT
