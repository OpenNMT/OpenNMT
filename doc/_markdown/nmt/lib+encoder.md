<a name="nmt.Encoder.dok"></a>


## nmt.Encoder ##

 Encoder is a unidirectional Sequencer used for the source language.

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [Sequencer](lib+sequencer).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/encoder.lua#L22">[src]</a>
<a name="nmt.Encoder"></a>


### nmt.Encoder(args, network) ###

 Constructor takes global `args` and optional `network`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/encoder.lua#L40">[src]</a>
<a name="nmt.Encoder:resize_proto"></a>


### nmt.Encoder:resize_proto(batch_size) ###

 Call to change the `batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/encoder.lua#L60">[src]</a>
<a name="nmt.Encoder:forward"></a>


### nmt.Encoder:forward(batch) ###

Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - last hidden states
  2. - context matrix H

TODO:

  * Change `batch` to `input`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/encoder.lua#L132">[src]</a>
<a name="nmt.Encoder:backward"></a>


### nmt.Encoder:backward(batch, grad_states_output, grad_context_output) ###

 Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `grad_states_output`
  * `grad_context_output` - gradient of loss
      wrt last states and context.

TODO: change this to (input, gradOutput) as in nngraph.



#### Undocumented methods ####

<a name="nmt.Encoder:convert"></a>
 * `nmt.Encoder:convert(f)`
