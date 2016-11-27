<a name="onmt.Sequencer.dok"></a>


## onmt.Sequencer ##

 Sequencer is the base class for encoder and decoder models.
  Main task is to manage `self.net(t)`, the unrolled network
  used during training.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/Sequencer.lua#L16">[src]</a>
<a name="onmt.Sequencer"></a>


### onmt.Sequencer(args, network) ###

 Constructor

Parameters:

  * `args` - global options.
  * `network` - optional recurrent step template.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/Sequencer.lua#L91">[src]</a>
<a name="onmt.Sequencer:net"></a>


### onmt.Sequencer:net(t) ###

Get access to the recurrent unit at a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/Sequencer.lua#L111">[src]</a>
<a name="onmt.Sequencer:training"></a>


### onmt.Sequencer:training() ###

 Move the network to train mode. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/Sequencer.lua#L121">[src]</a>
<a name="onmt.Sequencer:evaluate"></a>


### onmt.Sequencer:evaluate() ###

 Move the network to evaluation mode. 
