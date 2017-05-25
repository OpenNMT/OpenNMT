To prevent neural networks from overfitting and increases generalization capacity, several regularization methods are available.

## Dropout

Dropout is an extremely effective and simple regularization technique introduced by [Srivastava et al., 2015](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf). Dropout is only applied during training - and the idea is to disable for a given batch individual neurons with some probability \(p\). Setting \(p\) to 0 disables the dropout. Default regularization is dropout with \(p=0.2\).

![Dropout](../img/dropout.jpg)

!!! tip "Tip"
    Dropout value can be changed dynamically when restarting the training - so can be adjusted all along the training process.

Dropout applies on the output of each RNN layer, on the output of the attention layer, and can be enabled also between word embedding and the first layer with the option `-dropout_input`.

Because of recurrence, applying dropout to recurrent neural networks requires some specific care and two implementations are available and can be configured using `-dropout_type` option:

* `naive` (default): implements approach described in [Zaremba et al., 2015](https://arxiv.org/pdf/1409.2329.pdf). The dropout is only applied on non-recurrent connections.
* `variational`: implements approach described in [Gal et al., 2016]([https://arxiv.org/pdf/1512.05287.pdf]). In this approach, dropout is also applied to the recurrent connections but each timestep applies the same dropout mask.

The following picture (from Gal et al. paper) describes the different approaches. On the left side, the naive dropout: no dropout on recurrent connections, and dropout for each timestep is different. On the right side, the variational dropout: there is dropout on recurrent connections, but dropout for each timesteps are the same,

![Dropout Types](../img/dropout-type.jpg)

Finally dropout can also be applied to word embeddings, in that case, for each sentence in a minibatch, set of individual words randomly selected with probability \(p_{words}\) are replaced by a padding tokens. You can apply this option using `-dropout_words pwords` with a non zero value.

## Layer Normalization

Layer Normalization ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) is normalizing each neuron weighted input using the full layer for calculating \(\mu^{t,l}\) and \(\sigma^{t,l}\).

Layer Normalization is also including two learnt parameters per neuron: \(\textbf g_i\) which is the gain after normalization, and \(\textbf b_i\) which is the bias according to the formulas:

$$a'^{t,l}_i=\frac{\textbf g_i}{\sigma^{t,l}}.(a^{t,l}_i-\mu^{t,l})+{\textbf b_i}$$
$$\sigma^{t,l}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}(a^{t,l}_i-\mu^{t,l})^2}$$
$$\mu^{t,l}=\frac{1}{H}\sum_{i=1}^{H}a^{t,l}_i$$

Where: \(a^{t,l}\) is a neuron weighted input before activation in a layer \(l\) with \(H\) neuron at timestep \(t\).
