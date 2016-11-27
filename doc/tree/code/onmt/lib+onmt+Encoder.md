<a name="onmt.Encoder.dok"></a>


## onmt.Encoder ##

 Encoder is a unidirectional Sequencer used for the source language.

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [onmt.Sequencer](lib+onmt+sequencer).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Encoder.lua#L18">[src]</a>
<a name="onmt.Encoder"></a>


### onmt.Encoder(args, network) ###

 Constructor takes global `args` and optional `network`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Encoder.lua#L83">[src]</a>
<a name="onmt.Encoder:forward"></a>


### onmt.Encoder:forward(batch) ###

Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - last hidden states
  2. - context matrix H

TODO:

  * Change `batch` to `input`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Encoder.lua#L160">[src]</a>
<a name="onmt.Encoder:backward"></a>


### onmt.Encoder:backward(batch, grad_states_output, grad_context_output) ###

 Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `grad_states_output`
  * `grad_context_output` - gradient of loss
      wrt last states and context.

TODO: change this to (input, gradOutput) as in nngraph.

