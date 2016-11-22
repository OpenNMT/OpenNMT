<a name="nmt.EpochState.dok"></a>


## nmt.EpochState ##

 Class for managing the training process by logging and storing
  the state of the current epoch.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/train/epoch_state.lua#L9">[src]</a>
<a name="nmt.EpochState"></a>


### nmt.EpochState(epoch, status) ###

 Initialize for epoch `epoch` and training `status` (current loss)

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/train/epoch_state.lua#L28">[src]</a>
<a name="nmt.EpochState:update"></a>


### nmt.EpochState:update(batch, loss) ###

 Update training status. Takes `batch` (described in data.lua) and last loss.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/train/epoch_state.lua#L38">[src]</a>
<a name="nmt.EpochState:log"></a>


### nmt.EpochState:log(batch_index, data_size, learning_rate) ###

 Log to status stdout.
  TODO: these args shouldn't need to be passed in each time. 


#### Undocumented methods ####

<a name="nmt.EpochState:get_train_ppl"></a>
 * `nmt.EpochState:get_train_ppl()`
<a name="nmt.EpochState:get_time"></a>
 * `nmt.EpochState:get_time()`
<a name="nmt.EpochState:get_status"></a>
 * `nmt.EpochState:get_status()`
