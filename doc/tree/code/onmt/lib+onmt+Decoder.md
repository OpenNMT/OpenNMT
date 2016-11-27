<a name="onmt.Decoder.dok"></a>


## onmt.Decoder ##

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

Inherits from [onmt.Sequencer](lib+onmt+sequencer).



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L139">[src]</a>
<a name="onmt.Decoder:reset"></a>


### onmt.Decoder:reset(source_sizes, source_length, beam_size) ###

 Update internals to prepare for new batch.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L174">[src]</a>
<a name="onmt.Decoder:forward_one"></a>


### onmt.Decoder:forward_one(input, prev_states, context, prev_out, t) ###

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


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L214">[src]</a>
<a name="onmt.Decoder:forward_and_apply"></a>


### onmt.Decoder:forward_and_apply(batch, encoder_states, context, func) ###

Compute all forward steps.

  Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`
  * `func` - Calls `func(out, t)` each timestep.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L243">[src]</a>
<a name="onmt.Decoder:forward"></a>


### onmt.Decoder:forward(batch, encoder_states, context) ###

Compute all forward steps.

Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`

Returns:

  1. `outputs` - Top Hidden layer at each time-step.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L260">[src]</a>
<a name="onmt.Decoder:compute_score"></a>


### onmt.Decoder:compute_score(batch, encoder_states, context) ###

 Compute the cumulative score of a target sequence.
  Used in decoding when gold data are provided.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L276">[src]</a>
<a name="onmt.Decoder:compute_loss"></a>


### onmt.Decoder:compute_loss(batch, encoder_states, context, criterion) ###

 Compute the loss on a batch based on final layer `generator`.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/onmt/Decoder.lua#L292">[src]</a>
<a name="onmt.Decoder:backward"></a>


### onmt.Decoder:backward(batch, outputs, criterion) ###

 Compute the standard backward update.
  With input `batch`, target `outputs`, and `criterion`
  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.



#### Undocumented methods ####

<a name="onmt.Decoder"></a>
 * `onmt.Decoder(args, network, generator)`
