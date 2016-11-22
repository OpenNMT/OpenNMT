<a name="nmt.Decoder.dok"></a>


## nmt.Decoder ##

 Decoder is the sequencer for the target words.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [Sequencer](lib+sequencer).



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L124">[src]</a>
<a name="nmt.Decoder:resize_proto"></a>


### nmt.Decoder:resize_proto(batch_size) ###

 Call to change the `batch_size`. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L133">[src]</a>
<a name="nmt.Decoder:reset"></a>


### nmt.Decoder:reset(source_sizes, source_length, beam_size) ###

 Update internals to prepare for new batch.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L168">[src]</a>
<a name="nmt.Decoder:forward_one"></a>


### nmt.Decoder:forward_one(input, prev_states, context, prev_out, t) ###

 Run one step of the decoder.

Parameters:

 * `input` - sparse input (1)
 * `prev_states` - stack of hidden states (batch x layers*model x rnn_size)
 * `context` - encoder output (batch x n x rnn_size)
 * `prev_out` - previous distribution (batch x #words)
 * `t` - current timestep

Returns:

 1. `out` - Top-layer Hidden state
 2. `states` - All states


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L208">[src]</a>
<a name="nmt.Decoder:forward_and_apply"></a>


### nmt.Decoder:forward_and_apply(batch, encoder_states, context, func) ###

Compute all forward steps.

  Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`
  * `func` - Calls `func(out, t)` each timestep.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L231">[src]</a>
<a name="nmt.Decoder:forward"></a>


### nmt.Decoder:forward(batch, encoder_states, context) ###

Compute all forward steps.

Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`

Returns:

  1. `outputs` - Top Hidden layer at each time-step.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L263">[src]</a>
<a name="nmt.Decoder:compute_loss"></a>


### nmt.Decoder:compute_loss(batch, encoder_states, context, generator) ###

 Compute the loss on a batch based on final layer `generator`.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/decoder.lua#L279">[src]</a>
<a name="nmt.Decoder:backward"></a>


### nmt.Decoder:backward(batch, outputs, generator) ###

 Compute the standard backward update.
  With input `batch`, target `outputs`, and `generator`
  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.



#### Undocumented methods ####

<a name="nmt.Decoder"></a>
 * `nmt.Decoder(args, network)`
<a name="nmt.Decoder:compute_score"></a>
 * `nmt.Decoder:compute_score(batch, encoder_states, context, generator)`
<a name="nmt.Decoder:convert"></a>
 * `nmt.Decoder:convert(f)`
