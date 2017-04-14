During the training, some information are displayed every `-report_every` iterations. These logs are usually needed to evaluate the training progress, efficiency and convergence.

Measurements are reported as an average since the previous print.

## Perplexity

A key information is the **training perplexity** defined by:

$$ppl=\exp(\frac{loss}{|W|})$$

with \(loss\) being the cumulated negative log likelihood of the true target data and \(|W|\) the number of target words. You want this value to go down and be low in which case it means your model fits well the training data.

At the end of an epoch, the logs report the **validation perplexity** with the same formula but applied on the validation data. It shows how well your model fits unseen data.

!!! note "Note"
    During evaluation on the validation dataset, dropout is turned off.

## Logs management

Some advanced options are available to manage your logs like using a file (`-log_file`), or disabling them entirely (`-disable_logs`). See the options of the script to learn about them.
