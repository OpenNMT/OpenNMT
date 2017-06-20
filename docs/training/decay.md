OpenNMT's training implements empirical learning rate decay strategies. Experiences showed that using stochastic gradient descent (SGD) and a decay strategy yield better performance than optimization methods with adaptive learning rates.

Learning rate updates are always computed at the end of an epoch. When a decay condition is met, the following update rule is applied:

$$lr^{(t+1)} = lr^{(t)} \times decay$$

where \(lr^{(0)}=\) `-learning_rate` and \(decay=\) `-learning_rate_decay`.

If an epoch is a too large unit for your particular use case, consider using [data sampling](sampling.md). Additionally, it may be useful to set a minimum learning rate with `-min_learning_rate` to stop the training earlier when the learning rate is too small to make a difference.

## Default

By default, the decay is applied when one of the following conditions is met:

1. The validation score is not improving more than `-start_decay_score_delta`.
2. The current epoch is past `-start_decay_at`.

Once one of the conditions is met, the learning rate is decayed after **each** remaining epoch.

## Epoch-based

With the `-decay epoch_only` option, the learning rate is only decayed when the condition is met on the epoch:

1. The current epoch is past `-start_decay_at`.

## Score-based

With the `-decay score_only` option, the learning rate is only decayed when the condition is met on the validation score:

1. The validation score is not improving more than `-start_decay_score_delta`.
