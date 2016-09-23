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

This project is maintained by [Yoon Kim](http://people.fas.harvard.edu/~yoonkim).
Feel free to post any questions/issues on the issues page.

### Dependencies

#### Python
* h5py
* numpy

#### Lua
You will need the following packages:
* hdf5
* nn
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
th evaluate.lua -model demo-model_final.t7 -src_file data/src-val.txt -output_file pred.txt
-src_dict data/demo.src.dict -targ_dict data/demo.targ.dict
```
This will output predictions into `pred.txt`. The predictions are going to be quite terrible,
as the demo dataset is small. Try running on some larger datasets! For example you can download
millions of parallel sentences for [translation](http://www.statmt.org/wmt15/translation-task.html)
or [summarization](https://github.com/harvardnlp/sent-summary).

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
* `maxwordlength`: For the character models, words are truncated (if longer than maxwordlength)
or zero-padded (if shorter) to `maxwordlength`.
* `chars`: If 1, construct the character-level dataset as well.  This might take up a lot of space
depending on your data size, so you may want to break up the training data into different shards.
* `srcvocabfile, targetvocabfile`: If working with a preset vocab, then including these paths
will ignore the `srcvocabsize,targetvocabsize`.
* `unkfilter`: Ignore sentences with too many UNK tokens. Can be an absolute count limit (if > 1)
or a proportional limit (0 < unkfilter < 1).
* `shuffle`: Shuffle sentences.

#### Training options (`train.lua`)
**Data options**

* `data_file, val_data_file`: Path to the training/validation `*.hdf5` files created from running
`preprocess.py`.
* `savefile`: Savefile name (model will be saved as `savefile_epochX_PPL.t7` after every `save_every`
epoch where X is the X-th epoch and PPL is the validation perplexity at the epoch.
* `num_shards`: If the training data has been broken up into different shards,
then this is the number of shards.
* `train_from`: If training from a checkpoint then this is the path to the pre-trained model.

**Model options**

* `num_layers`: Number of layers in the LSTM encoder/decoder (i.e. number of stacks).
* `rnn_size`: Size of LSTM hidden states.
* `word_vec_size`: Word embedding size.
* `attn`:  If = 1, use attention over the source sequence during decoding. If = 0, then it
uses the last hidden state of the encoder as the context at each time step.
* `brnn`: If = 1, use a bidirectional LSTM on the encoder side. Input embeddings (or CharCNN
if using characters)  are shared between the forward/backward LSTM, and hidden states of the
corresponding forward/backward LSTMs are added to obtain the hidden representation for that
time step.
* `use_chars_enc`: If = 1, use characters on the encoder side (as inputs).
* `use_chars_dec`: If = 1, use characters on the decoder side (as inputs).
* `reverse_src`: If = 1, reverse the source sequence. The original sequence-to-sequence paper
found that this was crucial to achieving good performance, but with attention models this
does not seem necessary. Recommend leaving it to 0.
* `init_dec`: Initialize the hidden/cell state of the decoder at time 0 to be the last
hidden/cell state of the encoder. If 0, the initial states of the decoder are set to zero vectors.
* `input_feed`: If = 1, feed the context vector at each time step as additional input (via
concatenation with the word embeddings) to the decoder.
* `multi_attn`: If > 0, then use a *multi-attention* on this layer of the decoder. For example, if
`num_layers = 3` and `multi_attn = 2`, then the model will do an attention over the source sequence
on the second layer (and use that as input to the third layer) *and* the penultimate layer.
We've found that this did not really improve performance on translation, but may be helpful for
other tasks where multiple attentional passes over the source sequence are required
(e.g. for more complex reasoning tasks).
* `res_net`: Use residual connections between LSTM stacks whereby the input to the l-th LSTM
layer of the hidden state of the l-1-th LSTM layer summed with hidden state of the l-2th LSTM layer.
We didn't find this to really help in our experiments.

Below options only apply if using the character model.

* `char_vec_size`: If using characters, size of the character embeddings.
* `kernel_width`: Size (i.e. width) of the convolutional filter.
* `num_kernels`: Number of convolutional filters (feature maps). So the representation from characters will have this many dimensions.
* `num_highway_layers`: Number of highway layers in the character composition model.

**Optimization options**

* `epochs`: Number of training epochs.
* `start_epoch`: If loading from a checkpoint, the epoch from which to start.
* `param_init`: Parameters of the model are initialized over a uniform distribution with support
`(-param_init, param_init)`.
* `optim`: Optimization method, possible choices are 'sgd', 'adagrad', 'adadelta', 'adam'.
For seq2seq I've found vanilla SGD to work well but feel free to experiment.
* `learning_rate`: Starting learning rate. For 'adagrad', 'adadelta', and 'adam', this is the global
learning rate. Recommended settings vary based on `optim`: sgd (`learning_rate = 1`), adagrad
(`learning_rate = 0.1`), adadelta (`learning_rate = 1`), adam (`learning_rate = 0.1`).
* `max_grad_norm`: If the norm of the gradient vector exceeds this, renormalize to have its norm equal to `max_grad_norm`.
* `dropout`: Dropout probability. Dropout is applied between vertical LSTM stacks.
* `lr_decay`: Decay learning rate by this much if (i) perplexity does not decrease on the validation
set or (ii) epoch has gone past the `start_decay_at` epoch limit.
* `start_decay_at`: Start decay after this epoch.
* `curriculum`: For this many epochs, order the minibatches based on source sequence length. (Sometimes setting this to 1 will increase convergence speed).
* `feature_embeddings_dim_exponent`: If the additional feature takes `N` values, then the embbeding dimension will be set to `N^exponent`.
* `pre_word_vecs_enc`: If using pretrained word embeddings (on the encoder side), this is the
path to the *.hdf5 file with the embeddings. The hdf5 should have a single field `word_vecs`,
which references an array with dimensions vocab size by embedding size. Each row should be a word
embedding and follow the same indexing scheme as the *.dict files from running
`preprocess.py`. In order to be consistent with `beam.lua`, the first 4 indices should
always be `<blank>`, `<unk>`, `<s>`, `</s>` tokens.
* `pre_word_vecs_dec`: Path to *.hdf5 for pretrained word embeddings on the decoder side. See above
for formatting of the *.hdf5 file.
* `fix_word_vecs_enc`: If = 1, fix word embeddings on the encoder side.
* `fix_word_vecs_dec`: If = 1, fix word embeddings on the decoder side.
* `max_batch_l`: Batch size used to create the data in `preprocess.py`. If this is left blank
(recommended), then the batch size will be inferred from the validation set.

**Other options**

* `start_symbol`: Use special start-of-sentence and end-of-sentence tokens on the source side.
We've found this to make minimal difference.
* `gpuid`: Which GPU to use (-1 = use cpu).
* `gpuid2`: If this is >=0, then the model will use two GPUs whereby the encoder is on the first
GPU and the decoder is on the second GPU. This will allow you to train bigger models.
* `cudnn`: Whether to use cudnn or not for convolutions (for the character model). `cudnn`
has much faster convolutions so this is highly recommended if using the character model.
* `save_every`: Save every this many epochs.
* `print_every`: Print various stats after this many batches.
* `seed`: Change the random seed for random numbers in torch - use that option to train alternate models for ensemble
* `prealloc`: when set to 1 (default), enable memory preallocation and sharing between clones - this reduces by a lot the used memory - there should not be
any situation where you don't need it. Also - since memory is preallocated, there is not (major)
memory increase during the training. When set to 0, it rolls back to original memory optimization.

#### Decoding options (`beam.lua`)

* `model`: Path to model .t7 file.
* `src_file`: Source sequence to decode (one line per sequence).
* `targ_file`: True target sequence (optional).
* `output_file`: Path to output the predictions (each line will be the decoded sequence).
* `src_dict`: Path to source vocabulary (`*.src.dict` file from `preprocess.py`).
* `targ_dict`: Path to target vocabulary (`*.targ.dict` file from `preprocess.py`).
* `feature_dict_prefix`: Prefix of the path to the features vocabularies (`*.feature_N.dict` files from `preprocess.py`).
* `char_dict`: Path to character vocabulary (`*.char.dict` file from `preprocess.py`).
* `beam`: Beam size (recommend keeping this at 5).
* `max_sent_l`: Maximum sentence length. If any of the sequences in `srcfile` are longer than this
it will error out.
* `simple`: If = 1, output prediction is simply the first time the top of the beam
ends with an end-of-sentence token. If = 0, the model considers all hypotheses that have
been generated so far that ends with end-of-sentence token and takes the highest scoring
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
* `n_best`: If > 1, then it will also output an n_best list of decoded sentences in the following
format.
```
1 ||| sentence_1 ||| sentence_1_score
2 ||| sentence_2 ||| sentence_2_score
```
* `gpuid`: ID of the GPU to use (-1 = use CPU).
* `gpuid2`: ID if the second GPU (if specified).
* `cudnn`: If the model was trained with `cudnn`, then this should be set to 1 (otherwise the model
will fail to load).

#### Using additional input features
[Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/abs/1606.02892) (Senrich et al. 2016) shows that translation performance can be increased by using additional input features.

Similarly to this work, you can annotate each word in the **source** text by using the `-|-` separator:

```
word1-|-feat1-|-feat2 word2-|-feat1-|-feat2
```

It supports an arbitrary number of features with arbitrary labels. However, all input words must have the **same** number of annotations. See for example `data/src-train-case.txt` which annotates each word with the case information.

To evaluate the model, the option `-feature_dict_prefix` is required on `evaluate.lua` which points to the prefix of the features dictionnaries generated during the preprocessing.

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
