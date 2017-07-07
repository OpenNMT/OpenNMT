During the training, some information are displayed every `-report_every` iterations. These logs are usually needed to evaluate the training progress, efficiency and convergence.

!!! note "Note"
    Measurements are reported as an average since the previous print.

## Perplexity

A key information is the **training perplexity** defined by:

$$ppl(X,Y)=\exp(\frac{-\sum_{i=1}^{|Y|}\log P(y_i|y_{i-1},\dotsc,y_{1},X)}{|Y|})$$

with \(X\) being the source sequence, \(Y\) the true target sequence and \(y_i\) the \(i\)-th target word. The numerator is the negative log likelihood and the loss function value.

You want the perplexity to go down and be low in which case it means your model fits well the training data.

At the end of an epoch, the logs report by default the **validation perplexity** with the same formula but applied on the validation data. It shows how well your model fits unseen data. You can select other validation metrics with the `-validation_metric` option.

!!! note "Note"
    During evaluation on the validation dataset, dropout is turned off.

## Logs management

Some advanced options are available to manage your logs like using a file (`-log_file`), or disabling them entirely (`-disable_logs`). See the options of the script to learn about them.
