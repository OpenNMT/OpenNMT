<a name="onmt.EpochState.dok"></a>


## onmt.EpochState ##

 Class for managing the training process by logging and storing
  the state of the current epoch.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/train/epoch_state.lua#L9">[src]</a>
<a name="onmt.EpochState"></a>


### onmt.EpochState(epoch, num_iterations, learning_rate, status) ###

 Initialize for epoch `epoch` and training `status` (current loss)

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/train/epoch_state.lua#L30">[src]</a>
<a name="onmt.EpochState:update"></a>


### onmt.EpochState:update(batches, losses) ###

 Update training status. Takes `batch` (described in data.lua) and last losses.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/train/epoch_state.lua#L40">[src]</a>
<a name="onmt.EpochState:log"></a>


### onmt.EpochState:log(batch_index) ###

 Log to status stdout. 


#### Undocumented methods ####

<a name="onmt.EpochState:get_train_ppl"></a>
 * `onmt.EpochState:get_train_ppl()`
<a name="onmt.EpochState:get_time"></a>
 * `onmt.EpochState:get_time()`
<a name="onmt.EpochState:get_status"></a>
 * `onmt.EpochState:get_status()`
