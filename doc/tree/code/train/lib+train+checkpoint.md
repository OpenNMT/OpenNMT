<a name="onmt.Checkpoint.dok"></a>


## onmt.Checkpoint ##

Class for saving and loading models during training.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/a87c8c95a3cc254280aa661c2ffa86bca2bd7083/lib/train/checkpoint.lua#L34">[src]</a>
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
