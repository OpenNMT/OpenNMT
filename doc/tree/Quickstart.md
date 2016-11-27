<a id="onmt.Quickstart.Quickstart"></a>
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

