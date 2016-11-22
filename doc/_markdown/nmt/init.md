<a name="#OpenNMT.dok"/>

<a id="nmt.init.Style_Guide"></a>
<a id="nmt.init.Style_Guide"></a>
# Style Guide

<a id="nmt.init.Comments"></a>
<a id="nmt.init.Comments"></a>
## Comments 

* Comments should follow:
https://github.com/deepmind/torch-dokx/blob/master/doc/usage.md

* All non-private method should have dokx comments describing input/output.  

* All classes should have a class docstring at the top of the file. 

* All comments should be on their own line, and be a complete English
sentence with capitalization.

* Use torch-dokx and this command to build docs 
> dokx-build-package-docs -o docs .
> google-chrome doc/index.html


<a id="nmt.init.Style_"></a>
<a id="nmt.init.Style_"></a>
## Style:

* Please run and correct all warnings from luacheck before sending a pull request. 

> luacheck *

* All indentation should be 2 spaces.
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
<a id="nmt.init.Sequence_to_Sequence_Learning_with_Attentional_Neural_Networks"></a>
<a id="nmt.init.Sequence_to_Sequence_Learning_with_Attentional_Neural_Networks"></a>
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

<a id="nmt.init.Dependencies"></a>
<a id="nmt.init.Dependencies"></a>
### Dependencies

<a id="nmt.init.Lua"></a>
<a id="nmt.init.Lua"></a>
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

<a id="nmt.init.Quickstart"></a>
<a id="nmt.init.Quickstart"></a>
### Quickstart

We are going to be working with some example data in `data/` folder.
First run the data-processing code

```
th preprocess.lua -train_src_file data/src-train.txt -train_targ_file data/targ-train.txt
    -valid_src_file data/src-val.txt -valid_targ_file data/targ-val.txt -output_file data/demo
```

This will take the source/target train/valid files (`src-train.txt, targ-train.txt,
src-val.txt, targ-val.txt`) and build the following files:

* `demo.src.dict`: Dictionary of source vocab to index mappings.
* `demo.targ.dict`: Dictionary of target vocab to index mappings.
* `demo-train.t7`: serialized torch file containing vocabulary, training and validation data

The `*.dict` files are needed to check vocabulary, or to preprocess data with fixed vocabularies

Now run the model

```
th train.lua -data_file data/demo-train.t7 -savefile demo-model
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

<a id="nmt.init.Details"></a>
<a id="nmt.init.Details"></a>
### Details
<a id="nmt.init.Preprocessing_options__`preprocess_py`_"></a>
<a id="nmt.init.Preprocessing_options__`preprocess_py`_"></a>
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
* `alignfile`, `alignvalfile`: If provided with filenames that contain 'Pharaoh' format alignment
on the train and validation data, source-to-target alignments are stored in the dataset.

<a id="nmt.init.Training_options__`train_lua`_"></a>
<a id="nmt.init.Training_options__`train_lua`_"></a>
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

To build a model with guided alignment (implemented similarly to [Guided Alignment Training for Topic-Aware Neural Machine Translation](https://arxiv.org/abs/1607.01628) (Chen et al. 2016)):
* `guided_alignment`: If 1, use external alignments to guide the attention weights
* `guided_alignment_weight`: weight for guided alignment criterion
* `guided_alignment_decay`: decay rate per epoch for alignment weight

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
path to the file with the embeddings. The file should be a serialized Torch tensor with dimensions
vocab size by embedding size. Each row should be a word embedding and follow the same indexing
scheme as the *.dict files from running `preprocess.lua`. In order to be consistent with `beam.lua`,
the first 4 indices should always be `<blank>`, `<unk>`, `<s>`, `</s>` tokens.
* `pre_word_vecs_dec`: Path to the file for pretrained word embeddings on the decoder side. See above.
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

<a id="nmt.init.Decoding_options__`beam_lua`_"></a>
<a id="nmt.init.Decoding_options__`beam_lua`_"></a>
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
* `rescore`: when set to scorer name, use scorer to find hypothesis with highest score - available 'bleu', 'gleu'
* `rescore_param`: parameter to rescorer - for bleu/gleu ngram length

<a id="nmt.init.Using_additional_input_features"></a>
<a id="nmt.init.Using_additional_input_features"></a>
#### Using additional input features
[Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/abs/1606.02892) (Senrich et al. 2016) shows that translation performance can be increased by using additional input features.

Similarly to this work, you can annotate each word in the **source** text by using the `-|-` separator:

```
word1-|-feat1-|-feat2 word2-|-feat1-|-feat2
```

It supports an arbitrary number of features with arbitrary labels. However, all input words must have the **same** number of annotations. See for example `data/src-train-case.txt` which annotates each word with the case information.

To evaluate the model, the option `-feature_dict_prefix` is required on `evaluate.lua` which points to the prefix of the features dictionnaries generated during the preprocessing.

<a id="nmt.init.Pruning_a_model"></a>
<a id="nmt.init.Pruning_a_model"></a>
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

<a id="nmt.init.Switching_between_GPU_CPU_models"></a>
<a id="nmt.init.Switching_between_GPU_CPU_models"></a>
#### Switching between GPU/CPU models
By default, the model will always save the final model as a CPU model, but it will save the
intermediate models as a CPU/GPU model depending on how you specified `-gpuid`.
If you want to run beam search on the CPU with an intermediate model trained on the GPU,
you can use `convert_to_cpu.lua` to convert the model to CPU and run beam search.

<a id="nmt.init.GPU_memory_requirements_Training_speed"></a>
<a id="nmt.init.GPU_memory_requirements_Training_speed"></a>
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

<a id="nmt.init.Evaluation"></a>
<a id="nmt.init.Evaluation"></a>
#### Evaluation
For translation, evaluation via BLEU can be done by taking the output from `beam.lua` and using the
`multi-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder). For example

```
perl multi-bleu.perl gold.txt < pred.txt
```

<a id="nmt.init.Evaluation_of_States_and_Attention"></a>
<a id="nmt.init.Evaluation_of_States_and_Attention"></a>
#### Evaluation of States and Attention
attention_extraction.lua can be used to extract the attention and the LSTM states. It uses the following (required) options:

* `model`: Path to model .t7 file.
* `src_file`: Source sequence to decode (one line per sequence).
* `targ_file`: True target sequence.
* `src_dict`: Path to source vocabulary (`*.src.dict` file from `preprocess.py`).
* `targ_dict`: Path to target vocabulary (`*.targ.dict` file from `preprocess.py`).

Output of the script are two files, `encoder.hdf5` and `decoder.hdf5`. The encoder contains the states for every layer of the encoder LSTM and the offsets for the start of each source sentence. The decoder contains the states for the decoder LSTM layers and the offsets for the start of gold sentence. It additionally contains the attention for each time step (if the model uses attention).


<a id="nmt.init.Pre_trained_models"></a>
<a id="nmt.init.Pre_trained_models"></a>
#### Pre-trained models
We've uploaded English <-> German models trained on 4 million sentences from
[Workshop on Machine Translation 2015](http://www.statmt.org/wmt15/translation-task.html).
Download link is below:

https://drive.google.com/open?id=0BzhmYioWLRn_aEVnd0ZNcWd0Y2c

These models are 4-layer LSTMs with 1000 hidden units and essentially replicates the results from
[Effective Approaches to Attention-based
Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf),
Luong et al. EMNLP 2015.

<a id="nmt.init.Acknowledgments"></a>
<a id="nmt.init.Acknowledgments"></a>
#### Acknowledgments
Our implementation utilizes code from the following:
* [Andrej Karpathy's char-rnn repo](https://github.com/karpathy/char-rnn)
* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)
* [Element rnn library](https://github.com/Element-Research/rnn)

<a id="nmt.init.Licence"></a>
<a id="nmt.init.Licence"></a>
#### Licence
MIT
<a id="nmt.init.Highlight_js"></a>
<a id="nmt.init.Highlight_js"></a>
# Highlight.js

Highlight.js highlights syntax in code examples on blogs, forums and,
in fact, on any web page. It's very easy to use because it works
automatically: finds blocks of code, detects a language, highlights it.

Autodetection can be fine tuned when it fails by itself (see "Heuristics").


<a id="nmt.init.Basic_usage"></a>
<a id="nmt.init.Basic_usage"></a>
## Basic usage

Link the library and a stylesheet from your page and hook highlighting to
the page load event:

```html
<link rel="stylesheet" href="styles/default.css">
<script src="highlight.pack.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
```

This will highlight all code on the page marked up as `<pre><code> .. </code></pre>`.
If you use different markup or need to apply highlighting dynamically, read
"Custom initialization" below.

- You can download your own customized version of "highlight.pack.js" or
  use the hosted one as described on the download page:
  <http://highlightjs.org/download/>

- Style themes are available in the download package or as hosted files.
  To create a custom style for your site see the class reference in the file
  [classref.txt][cr] from the downloaded package.

[cr]: http://github.com/isagalaev/highlight.js/blob/master/classref.txt


<a id="nmt.init.node_js"></a>
<a id="nmt.init.node_js"></a>
## node.js

Highlight.js can be used under node.js. The package with all supported languages is
installable from NPM:

    npm install highlight.js

Alternatively, you can build it from the source with only languages you need:

    python3 tools/build.py -tnode lang1 lang2 ..

Using the library:

```javascript
var hljs = require('highlight.js');

// If you know the language
hljs.highlight(lang, code).value;

// Automatic language detection
hljs.highlightAuto(code).value;
```


<a id="nmt.init.AMD"></a>
<a id="nmt.init.AMD"></a>
## AMD

Highlight.js can be used with an AMD loader.  You will need to build it from
source in order to do so:

```bash
$ python3 tools/build.py -tamd lang1 lang2 ..
```

Which will generate a `build/highlight.pack.js` which will load as an AMD
module with support for the built languages and can be used like so:

```javascript
require(["highlight.js/build/highlight.pack"], function(hljs){

  // If you know the language
  hljs.highlight(lang, code).value;

  // Automatic language detection
  hljs.highlightAuto(code).value;
});
```


<a id="nmt.init.Tab_replacement"></a>
<a id="nmt.init.Tab_replacement"></a>
## Tab replacement

You can replace TAB ('\x09') characters used for indentation in your code
with some fixed number of spaces or with a `<span>` to give them special
styling:

```html
<script type="text/javascript">
  hljs.tabReplace = '    '; // 4 spaces
  // ... or
  hljs.tabReplace = '<span class="indent">\t</span>';

  hljs.initHighlightingOnLoad();
</script>
```

<a id="nmt.init.Custom_initialization"></a>
<a id="nmt.init.Custom_initialization"></a>
## Custom initialization

If you use different markup for code blocks you can initialize them manually
with `highlightBlock(code, tabReplace, useBR)` function. It takes a DOM element
containing the code to highlight and optionally a string with which to replace
TAB characters.

Initialization using, for example, jQuery might look like this:

```javascript
$(document).ready(function() {
  $('pre code').each(function(i, e) {hljs.highlightBlock(e)});
});
```

You can use `highlightBlock` to highlight blocks dynamically inserted into
the page. Just make sure you don't do it twice for already highlighted
blocks.

If your code container relies on `<br>` tags instead of line breaks (i.e. if
it's not `<pre>`) pass `true` into the third parameter of `highlightBlock`
to make highlight.js use `<br>` in the output:

```javascript
$('div.code').each(function(i, e) {hljs.highlightBlock(e, null, true)});
```


<a id="nmt.init.Heuristics"></a>
<a id="nmt.init.Heuristics"></a>
## Heuristics

Autodetection of a code's language is done using a simple heuristic:
the program tries to highlight a fragment with all available languages and
counts all syntactic structures that it finds along the way. The language
with greatest count wins.

This means that in short fragments the probability of an error is high
(and it really happens sometimes). In this cases you can set the fragment's
language explicitly by assigning a class to the `<code>` element:

```html
<pre><code class="html">...</code></pre>
```

You can use class names recommended in HTML5: "language-html",
"language-php". Classes also can be assigned to the `<pre>` element.

To disable highlighting of a fragment altogether use "no-highlight" class:

```html
<pre><code class="no-highlight">...</code></pre>
```


<a id="nmt.init.Export"></a>
<a id="nmt.init.Export"></a>
## Export

File export.html contains a little program that allows you to paste in a code
snippet and then copy and paste the resulting HTML code generated by the
highlighter. This is useful in situations when you can't use the script itself
on a site.


<a id="nmt.init.Meta"></a>
<a id="nmt.init.Meta"></a>
## Meta

- Version: 7.5
- URL:     http://highlightjs.org/

For the license terms see LICENSE files.
For authors and contributors see AUTHORS.en.txt file.
<a name="OpenNMT.dict.dok"></a>


<a id="nmt.init.OpenNMT_dict"></a>
<a id="nmt.init.OpenNMT_dict"></a>
## OpenNMT.dict ##



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.dict"></a>
 * `OpenNMT.dict(data)`
<a name="OpenNMT.dict:load_file"></a>
 * `OpenNMT.dict:load_file(filename)`
<a name="OpenNMT.dict:write_file"></a>
 * `OpenNMT.dict:write_file(filename)`
<a name="OpenNMT.dict:lookup"></a>
 * `OpenNMT.dict:lookup(key)`
<a name="OpenNMT.dict:set_special"></a>
 * `OpenNMT.dict:set_special(special)`
<a name="OpenNMT.dict:add_special"></a>
 * `OpenNMT.dict:add_special(label, idx)`
<a name="OpenNMT.dict:add_specials"></a>
 * `OpenNMT.dict:add_specials(labels)`
<a name="OpenNMT.dict:add"></a>
 * `OpenNMT.dict:add(label, idx)`
<a name="OpenNMT.dict:prune"></a>
 * `OpenNMT.dict:prune(size)`
<a name="OpenNMT.dict:convert_to_idx"></a>
 * `OpenNMT.dict:convert_to_idx(labels, start_symbols)`
<a name="OpenNMT.dict:convert_to_labels"></a>
 * `OpenNMT.dict:convert_to_labels(idx, stop)`
<a name="OpenNMT.Encoder.dok"></a>


<a id="nmt.init.OpenNMT_Encoder"></a>
<a id="nmt.init.OpenNMT_Encoder"></a>
## OpenNMT.Encoder ##

 Encoder is a unidirectional Sequencer used for the source language. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/encoder.lua#L9">[src]</a>
<a name="OpenNMT.Encoder"></a>


<a id="nmt.init.OpenNMT_Encoder_args__network_"></a>
<a id="nmt.init.OpenNMT_Encoder_args__network_"></a>
### OpenNMT.Encoder(args, network) ###

 Constructor takes global `args` and optional `network`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/encoder.lua#L27">[src]</a>
<a name="OpenNMT.Encoder:resize_proto"></a>


<a id="nmt.init.OpenNMT_Encoder_resize_proto_batch_size_"></a>
<a id="nmt.init.OpenNMT_Encoder_resize_proto_batch_size_"></a>
### OpenNMT.Encoder:resize_proto(batch_size) ###

 Call to change the `batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/encoder.lua#L47">[src]</a>
<a name="OpenNMT.Encoder:forward"></a>


<a id="nmt.init.OpenNMT_Encoder_forward_batch_"></a>
<a id="nmt.init.OpenNMT_Encoder_forward_batch_"></a>
### OpenNMT.Encoder:forward(batch) ###

Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - last hidden states
  2. - context matrix H

TODO:

  * Change `batch` to `input`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/encoder.lua#L119">[src]</a>
<a name="OpenNMT.Encoder:backward"></a>


<a id="nmt.init.OpenNMT_Encoder_backward_batch__grad_states_output__grad_context_output_"></a>
<a id="nmt.init.OpenNMT_Encoder_backward_batch__grad_states_output__grad_context_output_"></a>
### OpenNMT.Encoder:backward(batch, grad_states_output, grad_context_output) ###

 Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `grad_states_output`
  * `grad_context_output` - gradient of loss
      wrt last states and context.

TODO: change this to (input, gradOutput) as in nngraph.



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Encoder:convert"></a>
 * `OpenNMT.Encoder:convert(f)`
<a name="OpenNMT.BiEncoder.dok"></a>


<a id="nmt.init.OpenNMT_BiEncoder"></a>
<a id="nmt.init.OpenNMT_BiEncoder"></a>
## OpenNMT.BiEncoder ##

 BiEncoder is a bidirectional Sequencer used for the source language. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/biencoder.lua#L18">[src]</a>
<a name="OpenNMT.BiEncoder"></a>


<a id="nmt.init.OpenNMT_BiEncoder_args__merge__net_fwd__net_bwd_"></a>
<a id="nmt.init.OpenNMT_BiEncoder_args__merge__net_fwd__net_bwd_"></a>
### OpenNMT.BiEncoder(args, merge, net_fwd, net_bwd) ###

 Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.BiEncoder:resize_proto"></a>
 * `OpenNMT.BiEncoder:resize_proto(batch_size)`
<a name="OpenNMT.BiEncoder:forward"></a>
 * `OpenNMT.BiEncoder:forward(batch)`
<a name="OpenNMT.BiEncoder:backward"></a>
 * `OpenNMT.BiEncoder:backward(batch, grad_states_output, grad_context_output)`
<a name="OpenNMT.BiEncoder:training"></a>
 * `OpenNMT.BiEncoder:training()`
<a name="OpenNMT.BiEncoder:evaluate"></a>
 * `OpenNMT.BiEncoder:evaluate()`
<a name="OpenNMT.BiEncoder:convert"></a>
 * `OpenNMT.BiEncoder:convert(f)`


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Cuda.init"></a>
 * `OpenNMT.Cuda.init(opt)`
<a name="OpenNMT.Cuda.convert"></a>
 * `OpenNMT.Cuda.convert(obj)`
<a name="OpenNMT.EpochState.dok"></a>


<a id="nmt.init.OpenNMT_EpochState"></a>
<a id="nmt.init.OpenNMT_EpochState"></a>
## OpenNMT.EpochState ##

 Class for managing the training process by logging and storing
  the state of the current epoch.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/train/epoch_state.lua#L9">[src]</a>
<a name="OpenNMT.EpochState"></a>


<a id="nmt.init.OpenNMT_EpochState_epoch__status_"></a>
<a id="nmt.init.OpenNMT_EpochState_epoch__status_"></a>
### OpenNMT.EpochState(epoch, status) ###

 Initialize for epoch `epoch` and training `status` (current loss)

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/train/epoch_state.lua#L28">[src]</a>
<a name="OpenNMT.EpochState:update"></a>


<a id="nmt.init.OpenNMT_EpochState_update_batch__loss_"></a>
<a id="nmt.init.OpenNMT_EpochState_update_batch__loss_"></a>
### OpenNMT.EpochState:update(batch, loss) ###

 Update training status. Takes `batch` (described in data.lua) and last loss.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/train/epoch_state.lua#L38">[src]</a>
<a name="OpenNMT.EpochState:log"></a>


<a id="nmt.init.OpenNMT_EpochState_log_batch_index__data_size__learning_rate_"></a>
<a id="nmt.init.OpenNMT_EpochState_log_batch_index__data_size__learning_rate_"></a>
### OpenNMT.EpochState:log(batch_index, data_size, learning_rate) ###

 Log to status stdout.
  TODO: these args shouldn't need to be passed in each time. 


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.EpochState:get_train_ppl"></a>
 * `OpenNMT.EpochState:get_train_ppl()`
<a name="OpenNMT.EpochState:get_time"></a>
 * `OpenNMT.EpochState:get_time()`
<a name="OpenNMT.EpochState:get_status"></a>
 * `OpenNMT.EpochState:get_status()`


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Memory.optimize"></a>
 * `OpenNMT.Memory.optimize(model, batch)`

<a id="OpenNMT.STYLE.Style_Guide"></a>
<a id="nmt.init.Style_Guide"></a>
<a id="nmt.init.Style_Guide"></a>
# Style Guide

<a id="OpenNMT.STYLE.Comments"></a>
<a id="nmt.init.Comments"></a>
<a id="nmt.init.Comments"></a>
## Comments 

* Comments should follow:
https://github.com/deepmind/torch-dokx/blob/master/doc/usage.md

* All non-private method should have dokx comments describing input/output.  

* All classes should have a class docstring at the top of the file. 

* All comments should be on their own line, and be a complete English
sentence with capitalization.

* Use torch-dokx and this command to build docs 
> dokx-build-package-docs -o docs .
> google-chrome doc/index.html


<a id="OpenNMT.STYLE.Style_"></a>
<a id="nmt.init.Style_"></a>
<a id="nmt.init.Style_"></a>
## Style:

* Please run and correct all warnings from luacheck before sending a pull request. 

> luacheck *

* All indentation should be 2 spaces.
<a name="OpenNMT.Decoder.dok"></a>


<a id="nmt.init.OpenNMT_Decoder"></a>
<a id="nmt.init.OpenNMT_Decoder"></a>
## OpenNMT.Decoder ##

 Decoder is the sequencer for the target words.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L111">[src]</a>
<a name="OpenNMT.Decoder:resize_proto"></a>


<a id="nmt.init.OpenNMT_Decoder_resize_proto_batch_size_"></a>
<a id="nmt.init.OpenNMT_Decoder_resize_proto_batch_size_"></a>
### OpenNMT.Decoder:resize_proto(batch_size) ###

 Call to change the `batch_size`.

  TODO: rename.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L119">[src]</a>
<a name="OpenNMT.Decoder:reset"></a>


<a id="nmt.init.OpenNMT_Decoder_reset_source_sizes__source_length__beam_size_"></a>
<a id="nmt.init.OpenNMT_Decoder_reset_source_sizes__source_length__beam_size_"></a>
### OpenNMT.Decoder:reset(source_sizes, source_length, beam_size) ###

 Update internals to prepare for new batch.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L152">[src]</a>
<a name="OpenNMT.Decoder:forward_one"></a>


<a id="nmt.init.OpenNMT_Decoder_forward_one_input__prev_states__context__prev_out__t_"></a>
<a id="nmt.init.OpenNMT_Decoder_forward_one_input__prev_states__context__prev_out__t_"></a>
### OpenNMT.Decoder:forward_one(input, prev_states, context, prev_out, t) ###

 Run one step of the decoder.

Parameters:
 * `input` - sparse input (1)
 * `prev_states` - stack of hidden states (batch x layers*model x rnn_size)
 * `context` - encoder output (batch x n x rnn_size)
 * `prev_out` - previous distribution (batch x #words)
 * `t` - current timestep

Returns:
 1. `out` - Top-layer Hidden state
 2. `states` - All states


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L191">[src]</a>
<a name="OpenNMT.Decoder:forward_and_apply"></a>


<a id="nmt.init.OpenNMT_Decoder_forward_and_apply_batch__encoder_states__context__func_"></a>
<a id="nmt.init.OpenNMT_Decoder_forward_and_apply_batch__encoder_states__context__func_"></a>
### OpenNMT.Decoder:forward_and_apply(batch, encoder_states, context, func) ###

Compute all forward steps.

  Parameters:
  * `batch` - based on data.lua
  * `encoder_states`
  * `context`
  * `func` - Calls `func(out, t)` each timestep.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L212">[src]</a>
<a name="OpenNMT.Decoder:forward"></a>


<a id="nmt.init.OpenNMT_Decoder_forward_batch__encoder_states__context_"></a>
<a id="nmt.init.OpenNMT_Decoder_forward_batch__encoder_states__context_"></a>
### OpenNMT.Decoder:forward(batch, encoder_states, context) ###

Compute all forward steps.

Parameters:
  * `batch` - based on data.lua
  * `encoder_states`
  * `context`

Returns:
  1. `outputs` - Top Hidden layer at each time-step.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L244">[src]</a>
<a name="OpenNMT.Decoder:compute_loss"></a>


<a id="nmt.init.OpenNMT_Decoder_compute_loss_batch__encoder_states__context__generator_"></a>
<a id="nmt.init.OpenNMT_Decoder_compute_loss_batch__encoder_states__context__generator_"></a>
### OpenNMT.Decoder:compute_loss(batch, encoder_states, context, generator) ###

 Compute the loss on a batch based on final layer `generator`.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/decoder.lua#L263">[src]</a>
<a name="OpenNMT.Decoder:backward"></a>


<a id="nmt.init.OpenNMT_Decoder_backward_batch__outputs__generator_"></a>
<a id="nmt.init.OpenNMT_Decoder_backward_batch__outputs__generator_"></a>
### OpenNMT.Decoder:backward(batch, outputs, generator) ###

 Compute the standard backward update.
  With input `batch`, target `outputs`, and `generator`
  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.

  TODO: This object should own `generator` and or, generator should
  control its own backward pass.



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Decoder"></a>
 * `OpenNMT.Decoder(args, network)`
<a name="OpenNMT.Decoder:compute_score"></a>
 * `OpenNMT.Decoder:compute_score(batch, encoder_states, context, generator)`
<a name="OpenNMT.Decoder:convert"></a>
 * `OpenNMT.Decoder:convert(f)`
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
<a name="OpenNMT.Sequencer.dok"></a>


<a id="nmt.init.OpenNMT_Sequencer"></a>
<a id="nmt.init.OpenNMT_Sequencer"></a>
## OpenNMT.Sequencer ##

 Sequencer is the base class for our time series LSTM models.
  Acts similarly to an `nn.Module`.
   Main task is to manage `self.network_clones`, the unrolled LSTM
  used during training.
  Classes encoder/decoder/biencoder generalize these definitions.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/sequencer.lua#L207">[src]</a>
<a name="OpenNMT.Sequencer"></a>


<a id="nmt.init.OpenNMT_Sequencer_model__args__network_"></a>
<a id="nmt.init.OpenNMT_Sequencer_model__args__network_"></a>
### OpenNMT.Sequencer(model, args, network) ###

 Constructor

Parameters:
  * `model` - type of model (enc,dec)
  * `args` - global arguments
  * `network` - optional preconstructed network.

TODO: Should initialize all the members in this method.
   i.e. word_vecs, fix_word_vecs, network_clones, eval_mode, etc.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/sequencer.lua#L262">[src]</a>
<a name="OpenNMT.Sequencer:training"></a>


<a id="nmt.init.OpenNMT_Sequencer_training__"></a>
<a id="nmt.init.OpenNMT_Sequencer_training__"></a>
### OpenNMT.Sequencer:training() ###

 Tell the network to prepare for training mode. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/sequencer.lua#L301">[src]</a>
<a name="OpenNMT.Sequencer:evaluate"></a>


<a id="nmt.init.OpenNMT_Sequencer_evaluate__"></a>
<a id="nmt.init.OpenNMT_Sequencer_evaluate__"></a>
### OpenNMT.Sequencer:evaluate() ###

 Tell the network to prepare for evaluation mode. 


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Sequencer:resize_proto"></a>
 * `OpenNMT.Sequencer:resize_proto(batch_size)`
<a name="OpenNMT.Sequencer:backward_word_vecs"></a>
 * `OpenNMT.Sequencer:backward_word_vecs()`
<a name="OpenNMT.Sequencer:net"></a>
 * `OpenNMT.Sequencer:net(t)`
<a name="OpenNMT.Sequencer:convert"></a>
 * `OpenNMT.Sequencer:convert(f)`
<a name="OpenNMT.Optim.dok"></a>


<a id="nmt.init.OpenNMT_Optim"></a>
<a id="nmt.init.OpenNMT_Optim"></a>
## OpenNMT.Optim ##


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/train/optim.lua#L108">[src]</a>
<a name="OpenNMT.Optim:update_learning_rate"></a>


<a id="nmt.init.OpenNMT_Optim_update_learning_rate_score__epoch_"></a>
<a id="nmt.init.OpenNMT_Optim_update_learning_rate_score__epoch_"></a>
### OpenNMT.Optim:update_learning_rate(score, epoch) ###

decay learning rate if val perf does not improve or we hit the start_decay_at limit


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Optim"></a>
 * `OpenNMT.Optim(args)`
<a name="OpenNMT.Optim:update_params"></a>
 * `OpenNMT.Optim:update_params(params, grad_params, max_grad_norm)`
<a name="OpenNMT.Optim:get_learning_rate"></a>
 * `OpenNMT.Optim:get_learning_rate()`
<a name="OpenNMT.Optim:get_states"></a>
 * `OpenNMT.Optim:get_states()`
<a name="OpenNMT.Data.dok"></a>


<a id="nmt.init.OpenNMT_Data"></a>
<a id="nmt.init.OpenNMT_Data"></a>
## OpenNMT.Data ##

 Data management and batch creation.

Batch interface [size]: 

  * size: number of sentences in the batch [1]
  * source_length: max length in source batch [1]
  * source_size:  lengths of each source [batch x 1]
  * source_input:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * source_input_rev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * target_length: max length in source batch [1]
  * target_size: lengths of each source [batch x 1]
  * target_non_zeros: number of non-ignored words in batch [1]
  * target_input: input idx's of target (SABCDEPPPPPP) [batch x max]
  * target_output: expected output idx's of target (ABCDESPPPPPP) [batch x max]

 TODO: change name of size => maxlen


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/data.lua#L46">[src]</a>
<a name="OpenNMT.Data"></a>


<a id="nmt.init.OpenNMT_Data_src__targ_"></a>
<a id="nmt.init.OpenNMT_Data_src__targ_"></a>
### OpenNMT.Data(src, targ) ###

 Initialize a data object given aligned tables of IntTensors `src`
  and `targ`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/data.lua#L53">[src]</a>
<a name="OpenNMT.Data:set_batch_size"></a>


<a id="nmt.init.OpenNMT_Data_set_batch_size_max_batch_size_"></a>
<a id="nmt.init.OpenNMT_Data_set_batch_size_max_batch_size_"></a>
### OpenNMT.Data:set_batch_size(max_batch_size) ###

 Setup up the training data to respect `max_batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/data.lua#L103">[src]</a>
<a name="OpenNMT.Data:get_data"></a>


<a id="nmt.init.OpenNMT_Data_get_data_src__targ_"></a>
<a id="nmt.init.OpenNMT_Data_get_data_src__targ_"></a>
### OpenNMT.Data:get_data(src, targ) ###

 Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/data.lua#L158">[src]</a>
<a name="OpenNMT.Data:get_batch"></a>


<a id="nmt.init.OpenNMT_Data_get_batch_idx_"></a>
<a id="nmt.init.OpenNMT_Data_get_batch_idx_"></a>
### OpenNMT.Data:get_batch(idx) ###

 Get batch `idx`. If nil make a batch of all the data. 
<a id="OpenNMT.README.Sequence_to_Sequence_Learning_with_Attentional_Neural_Networks"></a>
<a id="nmt.init.Sequence_to_Sequence_Learning_with_Attentional_Neural_Networks"></a>
<a id="nmt.init.Sequence_to_Sequence_Learning_with_Attentional_Neural_Networks"></a>
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

<a id="OpenNMT.README.Dependencies"></a>
<a id="nmt.init.Dependencies"></a>
<a id="nmt.init.Dependencies"></a>
### Dependencies

<a id="OpenNMT.README.Lua"></a>
<a id="nmt.init.Lua"></a>
<a id="nmt.init.Lua"></a>
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

<a id="OpenNMT.README.Quickstart"></a>
<a id="nmt.init.Quickstart"></a>
<a id="nmt.init.Quickstart"></a>
### Quickstart

We are going to be working with some example data in `data/` folder.
First run the data-processing code

```
th preprocess.lua -train_src_file data/src-train.txt -train_targ_file data/targ-train.txt
    -valid_src_file data/src-val.txt -valid_targ_file data/targ-val.txt -output_file data/demo
```

This will take the source/target train/valid files (`src-train.txt, targ-train.txt,
src-val.txt, targ-val.txt`) and build the following files:

* `demo.src.dict`: Dictionary of source vocab to index mappings.
* `demo.targ.dict`: Dictionary of target vocab to index mappings.
* `demo-train.t7`: serialized torch file containing vocabulary, training and validation data

The `*.dict` files are needed to check vocabulary, or to preprocess data with fixed vocabularies

Now run the model

```
th train.lua -data_file data/demo-train.t7 -savefile demo-model
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

<a id="OpenNMT.README.Details"></a>
<a id="nmt.init.Details"></a>
<a id="nmt.init.Details"></a>
### Details
<a id="OpenNMT.README.Preprocessing_options__`preprocess_py`_"></a>
<a id="nmt.init.Preprocessing_options__`preprocess_py`_"></a>
<a id="nmt.init.Preprocessing_options__`preprocess_py`_"></a>
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
* `alignfile`, `alignvalfile`: If provided with filenames that contain 'Pharaoh' format alignment
on the train and validation data, source-to-target alignments are stored in the dataset.

<a id="OpenNMT.README.Training_options__`train_lua`_"></a>
<a id="nmt.init.Training_options__`train_lua`_"></a>
<a id="nmt.init.Training_options__`train_lua`_"></a>
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

To build a model with guided alignment (implemented similarly to [Guided Alignment Training for Topic-Aware Neural Machine Translation](https://arxiv.org/abs/1607.01628) (Chen et al. 2016)):
* `guided_alignment`: If 1, use external alignments to guide the attention weights
* `guided_alignment_weight`: weight for guided alignment criterion
* `guided_alignment_decay`: decay rate per epoch for alignment weight

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
path to the file with the embeddings. The file should be a serialized Torch tensor with dimensions
vocab size by embedding size. Each row should be a word embedding and follow the same indexing
scheme as the *.dict files from running `preprocess.lua`. In order to be consistent with `beam.lua`,
the first 4 indices should always be `<blank>`, `<unk>`, `<s>`, `</s>` tokens.
* `pre_word_vecs_dec`: Path to the file for pretrained word embeddings on the decoder side. See above.
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

<a id="OpenNMT.README.Decoding_options__`beam_lua`_"></a>
<a id="nmt.init.Decoding_options__`beam_lua`_"></a>
<a id="nmt.init.Decoding_options__`beam_lua`_"></a>
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
* `rescore`: when set to scorer name, use scorer to find hypothesis with highest score - available 'bleu', 'gleu'
* `rescore_param`: parameter to rescorer - for bleu/gleu ngram length

<a id="OpenNMT.README.Using_additional_input_features"></a>
<a id="nmt.init.Using_additional_input_features"></a>
<a id="nmt.init.Using_additional_input_features"></a>
#### Using additional input features
[Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/abs/1606.02892) (Senrich et al. 2016) shows that translation performance can be increased by using additional input features.

Similarly to this work, you can annotate each word in the **source** text by using the `-|-` separator:

```
word1-|-feat1-|-feat2 word2-|-feat1-|-feat2
```

It supports an arbitrary number of features with arbitrary labels. However, all input words must have the **same** number of annotations. See for example `data/src-train-case.txt` which annotates each word with the case information.

To evaluate the model, the option `-feature_dict_prefix` is required on `evaluate.lua` which points to the prefix of the features dictionnaries generated during the preprocessing.

<a id="OpenNMT.README.Pruning_a_model"></a>
<a id="nmt.init.Pruning_a_model"></a>
<a id="nmt.init.Pruning_a_model"></a>
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

<a id="OpenNMT.README.Switching_between_GPU_CPU_models"></a>
<a id="nmt.init.Switching_between_GPU_CPU_models"></a>
<a id="nmt.init.Switching_between_GPU_CPU_models"></a>
#### Switching between GPU/CPU models
By default, the model will always save the final model as a CPU model, but it will save the
intermediate models as a CPU/GPU model depending on how you specified `-gpuid`.
If you want to run beam search on the CPU with an intermediate model trained on the GPU,
you can use `convert_to_cpu.lua` to convert the model to CPU and run beam search.

<a id="OpenNMT.README.GPU_memory_requirements_Training_speed"></a>
<a id="nmt.init.GPU_memory_requirements_Training_speed"></a>
<a id="nmt.init.GPU_memory_requirements_Training_speed"></a>
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

<a id="OpenNMT.README.Evaluation"></a>
<a id="nmt.init.Evaluation"></a>
<a id="nmt.init.Evaluation"></a>
#### Evaluation
For translation, evaluation via BLEU can be done by taking the output from `beam.lua` and using the
`multi-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder). For example

```
perl multi-bleu.perl gold.txt < pred.txt
```

<a id="OpenNMT.README.Evaluation_of_States_and_Attention"></a>
<a id="nmt.init.Evaluation_of_States_and_Attention"></a>
<a id="nmt.init.Evaluation_of_States_and_Attention"></a>
#### Evaluation of States and Attention
attention_extraction.lua can be used to extract the attention and the LSTM states. It uses the following (required) options:

* `model`: Path to model .t7 file.
* `src_file`: Source sequence to decode (one line per sequence).
* `targ_file`: True target sequence.
* `src_dict`: Path to source vocabulary (`*.src.dict` file from `preprocess.py`).
* `targ_dict`: Path to target vocabulary (`*.targ.dict` file from `preprocess.py`).

Output of the script are two files, `encoder.hdf5` and `decoder.hdf5`. The encoder contains the states for every layer of the encoder LSTM and the offsets for the start of each source sentence. The decoder contains the states for the decoder LSTM layers and the offsets for the start of gold sentence. It additionally contains the attention for each time step (if the model uses attention).


<a id="OpenNMT.README.Pre_trained_models"></a>
<a id="nmt.init.Pre_trained_models"></a>
<a id="nmt.init.Pre_trained_models"></a>
#### Pre-trained models
We've uploaded English <-> German models trained on 4 million sentences from
[Workshop on Machine Translation 2015](http://www.statmt.org/wmt15/translation-task.html).
Download link is below:

https://drive.google.com/open?id=0BzhmYioWLRn_aEVnd0ZNcWd0Y2c

These models are 4-layer LSTMs with 1000 hidden units and essentially replicates the results from
[Effective Approaches to Attention-based
Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf),
Luong et al. EMNLP 2015.

<a id="OpenNMT.README.Acknowledgments"></a>
<a id="nmt.init.Acknowledgments"></a>
<a id="nmt.init.Acknowledgments"></a>
#### Acknowledgments
Our implementation utilizes code from the following:
* [Andrej Karpathy's char-rnn repo](https://github.com/karpathy/char-rnn)
* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)
* [Element rnn library](https://github.com/Element-Research/rnn)

<a id="OpenNMT.README.Licence"></a>
<a id="nmt.init.Licence"></a>
<a id="nmt.init.Licence"></a>
#### Licence
MIT
<a name="OpenNMT.Beam.dok"></a>


<a id="nmt.init.OpenNMT_Beam"></a>
<a id="nmt.init.OpenNMT_Beam"></a>
## OpenNMT.Beam ##

 Class for managing the beam search process. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/eval/beam.lua#L24">[src]</a>
<a name="OpenNMT.Beam"></a>


<a id="nmt.init.OpenNMT_Beam_size_"></a>
<a id="nmt.init.OpenNMT_Beam_size_"></a>
### OpenNMT.Beam(size) ###

Constructor

Parameters:
  * `size` : The beam `K`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/eval/beam.lua#L45">[src]</a>
<a name="OpenNMT.Beam:get_current_state"></a>


<a id="nmt.init.OpenNMT_Beam_get_current_state__"></a>
<a id="nmt.init.OpenNMT_Beam_get_current_state__"></a>
### OpenNMT.Beam:get_current_state() ###

 Get the outputs for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/eval/beam.lua#L50">[src]</a>
<a name="OpenNMT.Beam:get_current_origin"></a>


<a id="nmt.init.OpenNMT_Beam_get_current_origin__"></a>
<a id="nmt.init.OpenNMT_Beam_get_current_origin__"></a>
### OpenNMT.Beam:get_current_origin() ###

 Get the backpointers for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/eval/beam.lua#L63">[src]</a>
<a name="OpenNMT.Beam:advance"></a>


<a id="nmt.init.OpenNMT_Beam_advance_out__attn_out_"></a>
<a id="nmt.init.OpenNMT_Beam_advance_out__attn_out_"></a>
### OpenNMT.Beam:advance(out, attn_out) ###

 Given prob over words for every last beam `out` and attention
 `attn_out`. Compute and update the beam search.

Parameters:
  * `out`- probs at the last step
  * `attn_out`- attention at the last step

Returns: true if beam search is complete.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/eval/beam.lua#L127">[src]</a>
<a name="OpenNMT.Beam:get_hyp"></a>


<a id="nmt.init.OpenNMT_Beam_get_hyp_k_"></a>
<a id="nmt.init.OpenNMT_Beam_get_hyp_k_"></a>
### OpenNMT.Beam:get_hyp(k) ###

 Walk back to construct the full hypothesis `k`.

Parameters:
  * `k` - the position in the beam to construct.

Returns:
  1. The hypothesis
  2. The attention at each time step.



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Beam:sort_best"></a>
 * `OpenNMT.Beam:sort_best()`
<a name="OpenNMT.Beam:get_best"></a>
 * `OpenNMT.Beam:get_best()`
<a name="OpenNMT.file_reader.dok"></a>


<a id="nmt.init.OpenNMT_file_reader"></a>
<a id="nmt.init.OpenNMT_file_reader"></a>
## OpenNMT.file_reader ##



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.file_reader"></a>
 * `OpenNMT.file_reader(filename)`
<a name="OpenNMT.file_reader:next"></a>
 * `OpenNMT.file_reader:next()`
<a name="OpenNMT.file_reader:close"></a>
 * `OpenNMT.file_reader:close()`
<a name="OpenNMT.Model.dok"></a>


<a id="nmt.init.OpenNMT_Model"></a>
<a id="nmt.init.OpenNMT_Model"></a>
## OpenNMT.Model ##



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Model"></a>
 * `OpenNMT.Model()`
<a name="OpenNMT.Model:double"></a>
 * `OpenNMT.Model:double()`
<a name="OpenNMT.Model:float"></a>
 * `OpenNMT.Model:float()`
<a name="OpenNMT.Model:cuda"></a>
 * `OpenNMT.Model:cuda()`
<a name="OpenNMT.Generator.dok"></a>


<a id="nmt.init.OpenNMT_Generator"></a>
<a id="nmt.init.OpenNMT_Generator"></a>
## OpenNMT.Generator ##



<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Generator"></a>
 * `OpenNMT.Generator(args, network)`
<a name="OpenNMT.Generator:forward_one"></a>
 * `OpenNMT.Generator:forward_one(input)`
<a name="OpenNMT.Generator:training"></a>
 * `OpenNMT.Generator:training()`
<a name="OpenNMT.Generator:evaluate"></a>
 * `OpenNMT.Generator:evaluate()`
<a name="OpenNMT.Generator:convert"></a>
 * `OpenNMT.Generator:convert(f)`
<a name="OpenNMT.Checkpoint.dok"></a>


<a id="nmt.init.OpenNMT_Checkpoint"></a>
<a id="nmt.init.OpenNMT_Checkpoint"></a>
## OpenNMT.Checkpoint ##

Class for saving and loading models during training.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f4f15b7194da46fdd1d77a8cd0ec1183c320d562/lib/train/checkpoint.lua#L34">[src]</a>
<a name="OpenNMT.Checkpoint:save_iteration"></a>


<a id="nmt.init.OpenNMT_Checkpoint_save_iteration_iteration__epoch_state__batch_order_"></a>
<a id="nmt.init.OpenNMT_Checkpoint_save_iteration_iteration__epoch_state__batch_order_"></a>
### OpenNMT.Checkpoint:save_iteration(iteration, epoch_state, batch_order) ###

 Save the model and data in the middle of an epoch sorting the iteration. 


<a id="nmt.init.Undocumented_methods"></a>
<a id="nmt.init.Undocumented_methods"></a>
#### Undocumented methods ####

<a name="OpenNMT.Checkpoint"></a>
 * `OpenNMT.Checkpoint(args)`
<a name="OpenNMT.Checkpoint:save"></a>
 * `OpenNMT.Checkpoint:save(file_path, info)`
<a name="OpenNMT.Checkpoint:save_epoch"></a>
 * `OpenNMT.Checkpoint:save_epoch(valid_ppl, epoch_state)`
