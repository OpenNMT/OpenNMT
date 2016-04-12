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

* `srcvocabsize, targetvocabsize`: Size of source/target vocabularies. This is constructed
by taking the top X most frequent words. Rest are replaced with special UNK tokens.  
* `srcfile, targetfile`: Path to source/target training data, where each line represents a single
source/target sequence.  
* `srcvalfile, targetvalfile`: Path to source/target validation data.  
* `batchsize`: Size of each mini-batch.  
* `seqlength`: Maximum sequence length (sequences longer than this are dropped).  
* `outputfile`: Prefix of the output file names.  
* `maxwordlength`: For the character models, words are truncated (or zero-padded) to `maxwordlength`.  
* `chars`: If 1, construct the character-level dataset as well.  This might take up a lot of space
depending on your data size, so you may want to break up the training data into different shards.  
`srcvocabfile, targetvocabfile`: If working with a preset vocab, then including these paths
will ignore the `srcvocabsize,targetvocabsize`.  
* `unkfilter`: Ignore sentences with too many UNK tokens. Can be an absolute count limit (if > 1)
or a proportional limit (0 < unkfilter < 1).  

#### Training options (`train.lua`)
**Data options**

* `data_file, val_data_file`: Path to the training/validation `*.hdf5` files created from running
`preprocess.py`.  
* `savefile`: Savefile name (model will be saved as `savefile_epochX_PPL.t7` after every `save_every`
epoch where X is the X-th epoch and PPL is the validation perplexity at the epoch.  
* `num_shards`: If the training data has been broken up into different shards (by running
`preprocess-shards.py`), then this is the number of shards.  
* `train_from`: If training from a checkpoint then this is the path to the pre-trained model.  

**Model options**

* `num_layers`: Number of layers in the LSTM encoder/decoder (i.e. number of stacks).  
* `rnn_size`: Size of LSTM hidden states.  
* `word_vec_size`: Word embedding size.  
* `use_chars_enc`: If 1, use characters on the encoder side (as inputs).  
* `use_chars_dec`: If 1, use characters on the decoder side (as inputs).  
* `reverse_src`: If 1, reverse the source sequence. The original sequence-to-sequence paper
found that this was crucial to achieving good performance, but with attention models this
does not seem necessary. Recommend leaving it to 0.  
* `init_dec`: Inintialize the hidden/cell state of the decoder at time 0 to be the last
hidden/cell state of the encoder. If 0, the initial states of the decoder are set to zero vectors.  
* `hop_attn`: If > 0, then use a *hop attention* on this layer of the decoder. For example, if
`num_layers = 3` and `hop_attn = 2`, then the model will do an attention over the source sequence
on the second layer (and use that as input to the third layer) *and* the penultimate layer.
See [End-to-End Memory Networks](https://arxiv.org/abs/1503.08895) for more details. We've found that
this did not really improve performance on translation, but may be helpful for other tasks
where multiple attentional passes over the source sequence are required (e.g. for more complex
reasoning tasks).  

Below options only apply if using the character model.

* `char_vec_size`: If using characters, size of the character embeddings.  
* `kernel_width`: Size (i.e. width) of the convolutional filter.   
* `num_kernels`: Number of convolutional filters (feature maps). So the representation from characters
will have this many dimensions.  
* `num_highway_layers`: Number of highway layers in the character composition model.  

**Optimization options**

* `epochs`: Number of training epochs.  
* `start_epoch`: If loading from a pretrained model (or checkpoint), the epoch from which to
start at.  
* `param_init`: Parameters of the model are initialized over a uniform distribution with support
`(-param_init, param_init)`.  
* `learning_rate`: Starting learning rate.  
* `max_grad_norm`: If the norm of the gradient vector exceeds this, renormalize to have its norm equal
to `max_grad_norm`.  
* `dropout`: Dropout probability. Dropout is applied between vertical LSTM stacks.  
* `lr_decay`: Decay learning rate by this much if (i) perplexity does not decrease on the validation
set (ii) epoch has gone past the `start_decay_at` epoch limit.  
* `start_decay_at`: Start decay after this epoch.  
* `curriculum`: For this many epochs, order the training set based on source sequence length. (Sometimes setting this to 1 will increase convergence speed).  

**Other options**

* `start_symbol`: Use special start-of-sentence and end-of-sentence tokens in the source side.
We've found this to make minimal difference.    
* `gpuid`: Which gpu to use (-1 = use cpu).  
* `gpuid2`: If this is >=0, then the model will use two gpus whereby the encoder is on the first
gpu and the decoder is on the second gpu. This will allow you to train bigger models.  
* `cudnn`: Whether to use cudnn or not for convolutions (for the character model). `cudnn`
has much faster convolutions so this is highly recommended if using the character model.  
* `save_every`: Save every this many epochs.  
* `print_every`: Print various stats after this many batches.  
#### Decoding options (`beam.lua`)

* `modelfile`: Path to model .t7 file.  
* `srcfile`: Source sequence to decode (one line per sequence).  
* `targfile`: True target sequence (optional).  
* `outfile`: Path to output the predictions (each line will be the decoded sequence).  
* `srcdict`: Path to source vocabulary (`*.src.dict` file from `preprocess.py`).    
* `targdict`: Path to target vocabulary (`*.targ.dict` file from `preprocess.py`).    
* `chardict`: Path to character vocabulary (`*.char.dict` file from `preprocess.py`).    
* `beam`: Beam size (recommend keeping this at 5).    
* `max_sent_l`: Maximum sentence length. If any of the sequences in `srcfile` are longer than this
it will error out.    
* `simple`: If = 1, output prediction is simply the first time the top of the beam
ends with an end-of-sentence token. If = 0, the model considers all hypotheses that have
been generated so far and ends with end-of-sentence token and takes the highest scoring
of all of them.    
* `replace_unk`: Replace the generated UNK tokens with the source token that had the highest
attention weight. If `srctarg_dict` is provided, it will lookup the identified source token
and give the corresponding target token. If it is not provided (or the identified source token
does not exist in the table) then it will copy the source token.    
* `srctarg_dict`: Path to source-target dictionary to replace UNK tokens. Each line should be a
source token and its corresponding target token, separated by `|||`. For example
```
hello|||hallo
ukraine|||ukrainische
```
This dictionary can be obtained by, for example, running an alignment model as a preprocessing step.
We recommend [fast_align](https://github.com/clab/fast_align).  
* `score_gold`: If = 1, score the true target output as well.    
* `gpuid`: ID of the GPU to use.    
* `gpuid2`: ID if the second GPU (if specified).

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