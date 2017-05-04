To prevent neural networks from overfitting and increases generalization capacity, several regularization methods are available. Regularization applies on the output of each RNN layer, on the output of the attention layer, and can be enabled also between word embedding and the first layer with the option `-regularization_input`.

## Dropout

Dropout is an extremely effective and simple regularization techniques introduced by [Srivastava et al., 2015](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf). Dropout is only applied during training - and the idea is to disable for a given batch individual neurons with some probability \(p\). Setting \(p\) to 0 disables the dropout. Default regularization is dropout with \(p=0.2\).

![Dropout](http://cs231n.github.io/assets/nn2/dropout.jpeg)

!!! tip "Tip"
    Dropout value can be changed dynamically when restarting the training - so can be adjusted all along the training process.

## Layer Normalization

Layer Normalization [Ba et al., 2016](https://arxiv.org/abs/1607.06450) is normalizing each neuron output using the full layer for calculating \(\mu^l\) and \(\sigma^l\).
