OpenNMT's training implements empirical learning rate decay strategies. Experiences showed that using stochastic gradient descent (SGD) and a decay strategy yield better performance than optimization methods with adaptive learning rates.

Learning rate updates are always computed at the end of an epoch. When a decay condition is met, the following update rule is applied:

$$lr = lr \times decay$$

If an epoch is a too large unit for your particular use case, consider using [data sampling](sampling.md). Additionally, it may be useful to set a minimum learning rate with `-min_learning_rate` to stop the training earlier.

## Default

By default, the decay strategy is binary. There is either no decay or the decay is applied until the end of the training. The switch happens when one of the following condition is met first:

1. The validation perplexity is not improving more than `-start_decay_ppl_delta`
2. The current epoch is past `-start_decay_at`

## Perplexity-based

With the `-decay perplexity_only` option, learning rate is only decayed when the condition is met on the validation perplexity:

1. The validation perplexity is not improving more than `-start_decay_ppl_delta`
