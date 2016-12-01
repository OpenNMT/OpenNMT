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


Inherits from [onmt.Sequencer](lib+onmt+Sequencer).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Encoder.lua#L24">[src]</a>
<a name="onmt.Encoder"></a>


### onmt.Encoder(args, network) ###

 Construct an encoder layer.

Parameters:

  * `args` - global options.
  * `network` - optional recurrent step template.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Encoder.lua#L109">[src]</a>
<a name="onmt.Encoder:forward"></a>


### onmt.Encoder:forward(batch) ###

Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - final hidden states
  2. - context matrix H


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Encoder.lua#L190">[src]</a>
<a name="onmt.Encoder:backward"></a>


### onmt.Encoder:backward(batch, grad_states_output, grad_context_output) ###

 Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `grad_states_output` gradient of loss wrt last state
  * `grad_context_output` - gradient of loss wrt full context.

Returns: nil



#### Undocumented methods ####

<a name="onmt.Encoder:shareWordEmb"></a>
 * `onmt.Encoder:shareWordEmb(other)`
