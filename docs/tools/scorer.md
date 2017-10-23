OpenNMT provides native implementation of scoring metrics - BLEU, TER, DLRATIO

All metrics can be used as a validation metric (see [option `-validation_metric`](../options/train/#trainer-options)) during training or standalone using `tools/score.lua`:

```bash
$ th tools/score.lua REFERENCE [-sample SN] [-scorer bleu|ter|dlratio] PARAMS < OUT
```

The actual metric is selected with `scorer` option and the output is a line with 3 field, tab separated like:

```
34.73        +/-0.83        BLEU = 34.77, 79.8/49.1/29.6/17.6 (BP=0.919, ratio=0.922, hyp_len=26742, ref_len=28995)
54.77                       TER = 54.77 (Ins 1.8, Del 4.4, Sub 9.6, Shft 1.9, WdSh 2.6)
```

The fields are:

* numeric value of the score
* 95% confidence error margin (1.96*standard deviation) for k samples of half-size
* formated scorer output

!!! tip "Tip"
    *Error margin* is a simple way to know if score variation is part of metric calculation variation or is significant.

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

# TER

[TER](http://www.cs.umd.edu/~snover/tercom/) is an error metric for machine translation that messures the number of edits required to change a system output into one of the references. It is generally prefered to BLEU for estimation of sentence post-editing effort.

# DLRATIO

[Damerau-Levenshtein edit distance](https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance) is edit distance between 2 sentences. It is a simplified version of `TER` (in particular, `TER` that also integrates numbers of sequence shift).




