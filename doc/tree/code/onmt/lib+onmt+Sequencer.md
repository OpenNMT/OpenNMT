<a name="onmt.Sequencer.dok"></a>


## onmt.Sequencer ##

 Sequencer is the base class for encoder and decoder models.
  Main task is to manage `self.net(t)`, the unrolled network
  used during training.

     :net(1) => :net(2) => ... => :net(n-1) => :net(n)



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Sequencer.lua#L18">[src]</a>
<a name="onmt.Sequencer"></a>


### onmt.Sequencer(args, network) ###


Parameters:

  * `args` - global options.
  * `network` - recurrent step template.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Sequencer.lua#L88">[src]</a>
<a name="onmt.Sequencer:net"></a>


### onmt.Sequencer:net(t) ###

Get access to the recurrent unit at a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Sequencer.lua#L108">[src]</a>
<a name="onmt.Sequencer:training"></a>


### onmt.Sequencer:training() ###

 Move the network to train mode. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Sequencer.lua#L118">[src]</a>
<a name="onmt.Sequencer:evaluate"></a>


### onmt.Sequencer:evaluate() ###

 Move the network to evaluation mode. 
