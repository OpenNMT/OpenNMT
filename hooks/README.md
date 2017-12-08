This directory lists available hooks (see [documentation](http://opennmt.net/OpenNMT/misc/)) extending OpenNMT capability.

## SentencePiece

[SentencePiece](https://github.com/google/sentencepiece) is a sentence level tokenization implemented by Taku Kudo.

To use SentencePiece, you need to install sentencepiece and sentencepiece lua rock: see [here for detailed instructions](https://github.com/OpenNMT/lua-sentencepiece/blob/master/README.md).

Train models using `spm_train` and simply use it like that:

```
echo "It is a test-sample" | th tools/tokenize.lua -hook_file hooks/sentencepiece -sentencepiece myspmodel.mpdel -mode aggressive -joiner_annotate
```

Note that sentencepiece can be combined with regular tokenization - in that case, you do need to train the model on the same tokenization.

## Character tokenization

Simple character tokenization model.

```
echo "It is a test-sample" | th tools/tokenize.lua -hook_file hooks/chartokenization -mode char
I t ▁ i s ▁ a ▁ t e s t - s a m p l e
```

## Tree-Tagger Part-of-Speech annotation

This hook is interfacing with [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) and provides token pos annotation during tokenization.

To run it, you need to run a treetagger as a rest service using provided script:

```
python -u hooks/tree-tagger-server.py -model ~/french.par -path ~/bin/
```

which runs the REST service on the port 3000 of localhost.

and you can then use it during tokenization as follow:

```
th tools/tokenize.lua -hook_file hooks.tree-tagger -pos_feature < file
``` 