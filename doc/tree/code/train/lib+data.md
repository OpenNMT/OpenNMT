<a name="onmt.Data.dok"></a>


## onmt.Data ##

 Data management and batch creation.

Batch interface [size]:

  * size: number of sentences in the batch [1]
  * source_length: max length in source batch [1]
  * source_size:  lengths of each source [batch x 1]
  * source_input:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * source_input_features: table of source features sequences
  * source_input_rev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * source_input_rev_features: table of reversed source features sequences
  * target_length: max length in source batch [1]
  * target_size: lengths of each source [batch x 1]
  * target_non_zeros: number of non-ignored words in batch [1]
  * target_input: input idx's of target (SABCDEPPPPPP) [batch x max]
  * target_input_features: table of target input features sequences
  * target_output: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * target_output_features: table of target output features sequences

 TODO: change name of size => maxlen


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/data.lua#L51">[src]</a>
<a name="onmt.Data"></a>


### onmt.Data(src_data, targ_data) ###

 Initialize a data object given aligned tables of IntTensors `src`
  and `targ`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/data.lua#L63">[src]</a>
<a name="onmt.Data:set_batch_size"></a>


### onmt.Data:set_batch_size(max_batch_size) ###

 Setup up the training data to respect `max_batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/data.lua#L113">[src]</a>
<a name="onmt.Data:get_data"></a>


### onmt.Data:get_data(src, src_features, targ, targ_features) ###

 Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/data.lua#L203">[src]</a>
<a name="onmt.Data:get_batch"></a>


### onmt.Data:get_batch(idx) ###

 Get batch `idx`. If nil make a batch of all the data. 
