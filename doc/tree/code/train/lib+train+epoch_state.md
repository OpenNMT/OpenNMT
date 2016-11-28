<a name="onmt.EpochState.dok"></a>


## onmt.EpochState ##

 Class for managing the training process by logging and storing
  the state of the current epoch.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/train/epoch_state.lua#L9">[src]</a>
<a name="onmt.EpochState"></a>


### onmt.EpochState(epoch, status) ###

 Initialize for epoch `epoch` and training `status` (current loss)

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/train/epoch_state.lua#L28">[src]</a>
<a name="onmt.EpochState:update"></a>


### onmt.EpochState:update(batches, losses) ###

 Update training status. Takes `batch` (described in data.lua) and last losses.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/train/epoch_state.lua#L40">[src]</a>
<a name="onmt.EpochState:log"></a>


### onmt.EpochState:log(batch_index, data_size, learning_rate) ###

 Log to status stdout.
  TODO: these args shouldn't need to be passed in each time. 


#### Undocumented methods ####

<a name="onmt.EpochState:get_train_ppl"></a>
 * `onmt.EpochState:get_train_ppl()`
<a name="onmt.EpochState:get_time"></a>
 * `onmt.EpochState:get_time()`
<a name="onmt.EpochState:get_status"></a>
 * `onmt.EpochState:get_status()`
