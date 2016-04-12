## Sequence-to-Sequence Learning with Attentional Neural Networks

Implementation of a standard sequence-to-sequence model with attention where the encoder-decoder
are LSTMs. Also has the option to use characters (instead of input word embeddings)
by running a convolutional neural network followed by a highway network over
character embeddings to use as inputs.

The attention model is from
[Effective Approaches to Attention-based
Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf),
Luong et al. EMNLP 2015. We use the *global-attention* model with the *input-feeding* approach
from the paper.

The character model is from [Character-Aware Neural
Language Models](http://arxiv.org/abs/1508.06615), Kim et al. AAAI 2016.

### Dependencies

You will need the following packages:
* hdf5
* nngraph

GPU usage will additionally require:
* cutorch
* cunn

If running the character model, you should also install:
* cudnn
* luautf8

### Quickstart

We are going to be working with some example data in `data/` folder.
First run the data-processing code

```
python preprocess.py --srcfile data/src-train.txt --targetfile data/targ-train.txt
--srcvalfile data/src-val.txt --targetvalfile data/targ-val.txt --outputfile data/demo
```

This will take the source/target train/valid files (`src-train.txt, targ-train.txt,
src-val.txt, targ-val.txt`) and make some hdf5 files to be consumed by Lua.

`demo.src.dict`: Dictionary of source vocab to index mappings.  
`demo.targ.dict`: Dictionary of target vocab to index mappings.  
`demo-train.hdf5`: hdf5 containing the train data.  
`demo-val.hdf5`: hdf5 file containing the validation data.  

The `*.dict` files will be needed when predicting on new data.

Now run the model

```
th train.lua -data_file data/demo-train.hdf5 -val_data_file data/demo-val.hdf5 -savefile demo-model
```
This will run the default model, which consists of a 2-layer LSTM with 500 hidden units
on both the encoder/decoder.
You can also add `-gpuid 1` to use (say) GPU 1 in the cluster.

Now you have a model which you can use to predict on new data. To do this we are
going to be running beam search

```
th beam.lua -srcfile demo/src-train.txt -outfile pred.txt -srcdict demo/demo.src.dict
-targdict demo.targ.dict
```
This will output predictions into `pred.txt`. The predictions are going to be quite terrible,
as the demo dataset is small. Try running on some larger datasets! For example you can download
millions of parallel sentences for various language pairs from the [Workshop
on Machine Translation 2015](http://www.statmt.org/wmt15/translation-task.html).

### Details
#### Preprocessing options (`preprocess.py`)

$preprocessargs

#### Training options (`train.lua`)

$trainargs

#### Decoding options (`beam.lua`)

$beamargs

#### GPU memory requirements
Training large sequence-to-sequence models can be memory-intensive. Memory requirements will
dependend on batch size, maximum sequence length, vocabulary size, and (obviously) model size.
Here are some benchmark numbers (assuming batch size of 64, maximum sequence length of
50 on both the source/target sequence, and vocabulary size of 50000):

* 1-layer, 100 hidden units: 1.0G
* 1-layer, 250 hidden units: 1.5G
* 1-layer, 500 hidden units: 2.5G
* 2-layers, 500 hidden units: 3.2G
* 4-layers, 1000 hidden units: 8.8G
* 6-layers, 1000 hidden units: 11.5G

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

#### Pre-trained models
We will be uploading English <-> German models trained on 4 million sentences from
[Workshop on Machine Translation 2015](http://www.statmt.org/wmt15/translation-task.html).
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
