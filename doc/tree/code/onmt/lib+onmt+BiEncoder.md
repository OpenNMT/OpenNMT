<a name="onmt.BiEncoder.dok"></a>


## onmt.BiEncoder ##

 BiEncoder is a bidirectional Sequencer used for the source language.


 `net_fwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `net_bwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](lib+onmt+Sequencer).



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/BiEncoder.lua#L39">[src]</a>
<a name="onmt.BiEncoder"></a>


### onmt.BiEncoder(args, merge, net_fwd, net_bwd) ###

 Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).



#### Undocumented methods ####

<a name="onmt.BiEncoder:forward"></a>
 * `onmt.BiEncoder:forward(batch)`
<a name="onmt.BiEncoder:backward"></a>
 * `onmt.BiEncoder:backward(batch, grad_states_output, grad_context_output)`
