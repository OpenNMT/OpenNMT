OpenNMT provides native implementation of scoring metrics - for the moment, only BLEU.

# BLEU

[BLEU](https://en.wikipedia.org/wiki/BLEU) is a metric widely used for evaluation of machine translation output.

Syntax follows [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) syntax:

```bash
$ th tools/score.lua REFERENCE [-sample SN] [-scorer bleu] [-order N] < OUT
```

generating:
```log
[06/17/17 09:39:04 INFO] 4 references, 1002 sentences
BLEU = 34.77 +/- 0.43, 79.8/49.1/29.6/17.6 (BP=0.919, ratio=0.922, hyp_len=26742, ref_len=28995)
```

where:

* `REFERENCE` is either a single file, or a prefix for multiple-reference `REFERENCE0`, `REFERENCE1`, ...
* `-order` is bleu n-gram order (default 4)
* parameter `-sample` indicates the number of sample to draw for evaluating *error margin* (default 20) - indicated with `+/- ERR` in the output

!!! tip "Tip"
    *Error margin* is a simple way to know if BLEU score variation is part of metric calculation variation or is significant.


