By default, OpenNMT saves a checkpoint every 5000 iterations and at the end of each epoch. For more frequent or infrequent saves, you can use the `-save_every` and `-save_every_epochs` options which define the number of iterations and epochs after which the training saves a checkpoint.

There are several reasons one may want to train from a saved model with the `-train_from` option:

* continuing a stopped training
* continuing the training with a smaller batch size
* training a model on new data (incremental adaptation)
* starting a training from pre-trained parameters
* etc.

## Considerations

When training from an existing model, some settings can not be changed:

* the model topology (layers, hidden size, etc.)
* the vocabularies

!!! note "Exceptions"
    `-dropout`, `-fix_word_vecs_enc` and `-fix_word_vecs_dec` are model options that can be changed for a retraining.

## Resuming a stopped training

It is common that a training stops: crash, server reboot, user action, etc. In this case, you may want to continue the training for more epochs by using using the `-continue` flag. For example:

```bash
# start the initial training
th train.lua -gpuid 1 -data data/demo-train.t7 -save_model demo -save_every 50

# train for several epochs...

# need to reboot the server!

# continue the training from the last checkpoint
th train.lua -gpuid 1 -data data/demo-train.t7 -save_model demo -save_every 50 -train_from demo_checkpoint.t7 -continue
```

The `-continue` flag ensures that the training continues with the same configuration and optimization states. In particular, the following options are set to their last known value:

* `-curriculum`
* `-decay`
* `-learning_rate_decay`
* `-learning_rate`
* `-max_grad_norm`
* `-min_learning_rate`
* `-optim`
* `-start_decay_at`
* `-start_decay_ppl_delta`
* `-start_epoch`
* `-start_iteration`

!!! note "Note"
    The `-end_epoch` value is not automatically set as the user may want to continue its training for more epochs past the end.

Additionally, the `-continue` flag retrieves from the previous training:

* the non-SGD optimizers states
* the random generator states
* the batch order (when continuing from an intermediate checkpoint)

## Training from pre-trained parameters

Another use case it to use a base model and train it further with new training options (in particular the optimization method and the learning rate). Using `-train_from` without `-continue` will start a new training with parameters initialized from a pre-trained model.

## Updating the vocabularies
* `-update_vocab <string>` (accepted: `none`, `replace`, `merge`; default: `none`)

It is possible that we restart the training with a new dataset such as dynamic dataset, we could have different vocabularies in dynamic dataset and the pre-trained model. Instead of re-initializing the whole network, the pre-trained states of the common words in the new/previous dictionaries can be kept with option `-update_vocab`. This option is disabled by default and the update of word features isn't supported for instant.
`replace` mode will only keep the common words. For non-common words, the old ones will be deleted and the new onse will be initialized.
`merge` mode will keep the state of all the old words. The new words will be initialized.
