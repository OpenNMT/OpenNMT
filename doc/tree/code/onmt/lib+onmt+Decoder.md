<a name="onmt.Decoder.dok"></a>


## onmt.Decoder ##

 Unit to decode a sequence of output tokens.

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

Inherits from [onmt.Sequencer](lib+onmt+Sequencer).



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L28">[src]</a>
<a name="onmt.Decoder"></a>


### onmt.Decoder(args, network, generator) ###

 Construct an encoder layer.

Parameters:

  * `args` - global options.
  * `network` - optional, recurrent step template.
  * `generator` - optional, a output [onmt.Generator](lib+onmt+Generator).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L176">[src]</a>
<a name="onmt.Decoder:reset"></a>


### onmt.Decoder:reset(source_sizes, source_length, beam_size) ###

 Update internals of model to prepare for new batch.

  Parameters:

  * See  [onmt.MaskedSoftmax](lib+onmt+MaskedSoftmax).


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L212">[src]</a>
<a name="onmt.Decoder:forward_one"></a>


### onmt.Decoder:forward_one(input, features, prev_states, context, prev_out, t) ###

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


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L289">[src]</a>
<a name="onmt.Decoder:forward"></a>


### onmt.Decoder:forward(batch, encoder_states, context) ###

Compute all forward steps.

Parameters:

  * `batch` - based on data.lua
  * `encoder_states` - the final encoder states
  * `context` - the context to apply attention to.

Returns: Tables of top hidden layer at each timestep.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L306">[src]</a>
<a name="onmt.Decoder:compute_score"></a>


### onmt.Decoder:compute_score(batch, encoder_states, context) ###

 Compute the cumulative score of a target sequence.
  Used in decoding when gold data are provided.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L322">[src]</a>
<a name="onmt.Decoder:compute_loss"></a>


### onmt.Decoder:compute_loss(batch, encoder_states, context, criterion) ###

 Compute the loss on a batch based on final layer `generator`.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/Decoder.lua#L350">[src]</a>
<a name="onmt.Decoder:backward"></a>


### onmt.Decoder:backward(batch, outputs, criterion) ###

 Compute the standard backward update.

Parameters:

  * `batch`
  * `outputs`
  * `criterion`

  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.



#### Undocumented methods ####

<a name="onmt.Decoder:forward_and_apply"></a>
 * `onmt.Decoder:forward_and_apply(batch, encoder_states, context, func)`
