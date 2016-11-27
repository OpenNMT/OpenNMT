<a name="onmt.Sequencer.dok"></a>


## onmt.Sequencer ##

 Sequencer is the base class for encoder and decoder models.
  Main task is to manage `self.network_clones`, the unrolled network
  used during training.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Sequencer.lua#L17">[src]</a>
<a name="onmt.Sequencer"></a>


### onmt.Sequencer(args, network) ###

 Constructor

Parameters:

  * `args` - global arguments
  * `network` - network to unroll.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Sequencer.lua#L92">[src]</a>
<a name="onmt.Sequencer:net"></a>


### onmt.Sequencer:net(t) ###

Get a clone for a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Sequencer.lua#L112">[src]</a>
<a name="onmt.Sequencer:training"></a>


### onmt.Sequencer:training() ###

 Tell the network to prepare for training mode. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Sequencer.lua#L122">[src]</a>
<a name="onmt.Sequencer:evaluate"></a>


### onmt.Sequencer:evaluate() ###

 Tell the network to prepare for evaluation mode. 
