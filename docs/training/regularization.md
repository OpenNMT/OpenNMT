To prevent neural networks from overfitting and increases generalization capacity, several regularization methods are available.

## Dropout

Dropout is an extremely effective and simple regularization techniques introduced by [Srivastava et al., 2015](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf). Dropout is only applied during training - and the idea is to disable for a given batch individual neurons with some probability \(p\). Setting \(p\) to 0 disables the dropout. Default regularization is dropout with \(p=0.2\).

![Dropout](http://cs231n.github.io/assets/nn2/dropout.jpeg)

!!! tip "Tip"
    Dropout value can be changed dynamically when restarting the training - so can be adjusted all along the training process.

Dropout applies on the output of each RNN layer, on the output of the attention layer, and can be enabled also between word embedding and the first layer with the option `-dropout_input`.

## Layer Normalization

Layer Normalization [Ba et al., 2016](https://arxiv.org/abs/1607.06450) is normalizing each neuron weighted input using the full layer for calculating \(\mu^{t,l}\) and \(\sigma^{t,l}\).

Layer Normalization is also including two learnt parameters per neuron: \(\textbf g_i\) which the gain after normalization, and \(\textbf b_i\) which is the bias according to the formulas:

$$h'^{t,l}_i=\frac{\textbf g_i}{\sigma^{t,l}}.(h^{t,l}_i-\mu^{t,l})+{\textbf b_i}$$
$$\sigma^{t,l}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}(h^{t,l}_i-\mu^{t,l})^2}$$
$$\mu^{t,l}=\frac{1}{H}\sum_{i=1}^{H}h^{t,l}_i$$
