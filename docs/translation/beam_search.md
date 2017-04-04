By default, translation is done using beam search. The `-beam_size` option can be used to trade-off translation time and search accuracy, with `-beam_size 1` giving greedy search. The small default beam size is often enough in practice.

Beam search can also be used to provide an approximate n-best list of translations by setting `-n_best` greater than 1. For analysis, the translation command also takes an oracle/gold `-tgt` file and will output a comparison of scores.

## Hypotheses filtering

The beam search provides a built-in filter based on unknown words: `-max_num_unks`. Hypotheses with for more unkown words than this value are dropped.

!!! note "Note"
    As dropped hypotheses temporarily reduce the beam size, the `-pre_filter_factor` is a way to increase the number of considered hypotheses before applying filters.

## Normalization

The beam search also supports various normalization techniques that are disabled by default.

### Length normalization

Scores are normalized by the following formula as defined in [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf):

$$lp(Y) = \frac{(5+|Y|)^\alpha}{(5+1)^\alpha}$$

where \(|Y|\) is the current target length and \(\alpha\) is the length normalization coefficient `-length_norm`.

### Coverage normalization

Scores are penalized by the following formula as defined in [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf):

$$cp(X; Y) = \beta \times \sum_{i=1}^{|X|}\log(\min(\sum_{j=1}^{|Y|}p_{i,j},1.0))$$

where \(p_{i,j}\) is the attention probability of the \(j\)-th target word \(y_j\) on the \(i\)-th source word \(x_i\), \(|X|\) is the source length, \(|Y|\) is the current target length and \(\beta\) is the coverage normalization coefficient `-coverage_norm`.

### End of sentence normalization

The score of the end of sentence token is penalized by the following formula:

$$\frac{|X|}{|Y|}*\gamma$$

where \(|X|\) is the source length, \(|Y|\) is the current target length and \(\gamma\) is the coverage normalization coefficient `-eos_norm`.

This can be used to make translation longer.
