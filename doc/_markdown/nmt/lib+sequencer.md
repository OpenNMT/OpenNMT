<a name="nmt.Sequencer.dok"></a>


## nmt.Sequencer ##

 Sequencer is the base class for our time series LSTM models.
  Acts similarly to an `nn.Module`.
   Main task is to manage `self.network_clones`, the unrolled LSTM
  used during training.
  Classes encoder/decoder/biencoder generalize these definitions.

  Inherits from [Model](lib+model).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/sequencer.lua#L213">[src]</a>
<a name="nmt.Sequencer"></a>


### nmt.Sequencer(model, args, network) ###

 Constructor

Parameters:

  * `model` - type of model (enc,dec)
  * `args` - global arguments
  * `network` - optional preconstructed network.

TODO: Should initialize all the members in this method.
   i.e. word_vecs, fix_word_vecs, network_clones, eval_mode, etc.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/sequencer.lua#L268">[src]</a>
<a name="nmt.Sequencer:training"></a>


### nmt.Sequencer:training() ###

 Tell the network to prepare for training mode. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/sequencer.lua#L307">[src]</a>
<a name="nmt.Sequencer:evaluate"></a>


### nmt.Sequencer:evaluate() ###

 Tell the network to prepare for evaluation mode. 


#### Undocumented methods ####

<a name="nmt.Sequencer:resize_proto"></a>
 * `nmt.Sequencer:resize_proto(batch_size)`
<a name="nmt.Sequencer:backward_word_vecs"></a>
 * `nmt.Sequencer:backward_word_vecs()`
<a name="nmt.Sequencer:net"></a>
 * `nmt.Sequencer:net(t)`
<a name="nmt.Sequencer:convert"></a>
 * `nmt.Sequencer:convert(f)`
