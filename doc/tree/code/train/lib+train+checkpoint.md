<a name="onmt.Checkpoint.dok"></a>


## onmt.Checkpoint ##

Class for saving and loading models during training.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/train/checkpoint.lua#L31">[src]</a>
<a name="onmt.Checkpoint:save_iteration"></a>


### onmt.Checkpoint:save_iteration(iteration, epoch_state, batch_order) ###

 Save the model and data in the middle of an epoch sorting the iteration. 


#### Undocumented methods ####

<a name="onmt.Checkpoint"></a>
 * `onmt.Checkpoint(options, nets, optim, dataset)`
<a name="onmt.Checkpoint:save"></a>
 * `onmt.Checkpoint:save(file_path, info)`
<a name="onmt.Checkpoint:save_epoch"></a>
 * `onmt.Checkpoint:save_epoch(valid_ppl, epoch_state)`
