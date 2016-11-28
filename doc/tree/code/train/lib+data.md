<a name="onmt.Data.dok"></a>


## onmt.Data ##

 Data management and batch creation.

Batch interface [size]:

  * size: number of sentences in the batch [1]
  * source_length: max length in source batch [1]
  * source_size:  lengths of each source [batch x 1]
  * source_input:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * source_input_rev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * target_length: max length in source batch [1]
  * target_size: lengths of each source [batch x 1]
  * target_non_zeros: number of non-ignored words in batch [1]
  * target_input: input idx's of target (SABCDEPPPPPP) [batch x max]
  * target_output: expected output idx's of target (ABCDESPPPPPP) [batch x max]

 TODO: change name of size => maxlen


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/data.lua#L47">[src]</a>
<a name="onmt.Data"></a>


### onmt.Data(src, targ) ###

 Initialize a data object given aligned tables of IntTensors `src`
  and `targ`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/data.lua#L54">[src]</a>
<a name="onmt.Data:set_batch_size"></a>


### onmt.Data:set_batch_size(max_batch_size) ###

 Setup up the training data to respect `max_batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/data.lua#L104">[src]</a>
<a name="onmt.Data:get_data"></a>


### onmt.Data:get_data(src, targ, nocuda) ###

 Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/data.lua#L164">[src]</a>
<a name="onmt.Data:get_batch"></a>


### onmt.Data:get_batch(idx, nocuda) ###

 Get batch `idx`. If nil make a batch of all the data. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/data.lua#L195">[src]</a>
<a name="onmt.Data:distribute"></a>


### onmt.Data:distribute(batch, count) ###

 Slice batch into several smaller batcher for data parallelism. 
