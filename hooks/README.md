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

## TreeTagger Part-of-Speech annotation

This hook is interfacing with [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) and provides POS and/or lemma annotation during tokenization.

First start TreeTagger as a REST service using the provided script:

```
python -u hooks/tree-tagger-server.py -model /TREE-TAGGER-lib-dir/your-language.par -path /TREE-TAGGER-bin-dir/
```

This runs the REST service on port 3000 of localhost.
To see all available options, just type:
```
python hooks/tree-tagger-server.py -h
```

Once the REST server is running, you can use it during tokenization as follows:

```
th tools/tokenize.lua -hook_file hooks.tree-tagger -pos_feature < file
``` 
