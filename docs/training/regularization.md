To prevent neural networks from overfitting and increase generalization capacity, several regularization methods are available.

## Dropout

Dropout is an effective and simple regularization technique introduced by [Srivastava et al., 2015](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf). Dropout is only applied during training. The idea is to disable for a given batch individual neurons with some probability \(p\). Setting \(p\) to 0 disables the dropout. Default regularization is dropout with \(p=0.2\).

![Dropout](../img/dropout.jpg)

!!! tip "Tip"
    Dropout value can be changed dynamically when restarting the training. So it can be adjusted all along the training process.

Dropout is applied on the output of each layer, the output of the attention layer, and can be enabled also between word embeddings and the first layer with the `-dropout_input` option.

Because of recurrence, applying dropout to recurrent neural networks requires some specific care and two implementations are available and can be configured using the `-dropout_type` option:

* `naive` (default): implements the approach described in [Zaremba et al., 2015](https://arxiv.org/pdf/1409.2329.pdf). The dropout is only applied on non-recurrent connections.
* `variational`: implements the approach described in [Gal et al., 2016](https://arxiv.org/pdf/1512.05287.pdf). In this approach, dropout is also applied to the recurrent connections but each timestep applies the same dropout mask.
* `variational_non_recurrent`: hybrid approach - each timestep the same dropout mask, but no dropout on the recurrent connections

The following picture (from Gal et al. paper) describes both different approaches. On the left side, the naive dropout: no dropout on recurrent connections, and dropout for each timestep is different. On the right side, the variational dropout: there is dropout on recurrent connections, but dropout for each timesteps are the same,

![Dropout Types](../img/dropout-type.jpg)

Finally, dropout can also be applied to the words themselves. In that case, a set of individual words randomly selected with probability \(p_{words}\) are replaced by padding tokens. You can apply this option using `-dropout_words pwords` with a non zero value.
