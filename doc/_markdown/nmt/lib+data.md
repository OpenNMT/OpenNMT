<a name="nmt.Data.dok"></a>


## nmt.Data ##

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


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/data.lua#L46">[src]</a>
<a name="nmt.Data"></a>


### nmt.Data(src, targ) ###

 Initialize a data object given aligned tables of IntTensors `src`
  and `targ`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/data.lua#L53">[src]</a>
<a name="nmt.Data:set_batch_size"></a>


### nmt.Data:set_batch_size(max_batch_size) ###

 Setup up the training data to respect `max_batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/data.lua#L103">[src]</a>
<a name="nmt.Data:get_data"></a>


### nmt.Data:get_data(src, targ) ###

 Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/data.lua#L158">[src]</a>
<a name="nmt.Data:get_batch"></a>


### nmt.Data:get_batch(idx) ###

 Get batch `idx`. If nil make a batch of all the data. 
