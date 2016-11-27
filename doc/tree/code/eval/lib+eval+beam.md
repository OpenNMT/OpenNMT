<a name="onmt.Beam.dok"></a>


## onmt.Beam ##

 Class for managing the beam search process. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/eval/beam.lua#L26">[src]</a>
<a name="onmt.Beam"></a>


### onmt.Beam(size) ###

Constructor

Parameters:

  * `size` : The beam `K`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/eval/beam.lua#L47">[src]</a>
<a name="onmt.Beam:get_current_state"></a>


### onmt.Beam:get_current_state() ###

 Get the outputs for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/eval/beam.lua#L52">[src]</a>
<a name="onmt.Beam:get_current_origin"></a>


### onmt.Beam:get_current_origin() ###

 Get the backpointers for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/eval/beam.lua#L66">[src]</a>
<a name="onmt.Beam:advance"></a>


### onmt.Beam:advance(out, attn_out) ###

 Given prob over words for every last beam `out` and attention
 `attn_out`. Compute and update the beam search.

Parameters:

  * `out`- probs at the last step
  * `attn_out`- attention at the last step

Returns: true if beam search is complete.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/71aa250c35a20c1cf83f5f0150c1f900dc11d096/lib/eval/beam.lua#L132">[src]</a>
<a name="onmt.Beam:get_hyp"></a>


### onmt.Beam:get_hyp(k) ###

 Walk back to construct the full hypothesis `k`.

Parameters:

  * `k` - the position in the beam to construct.

Returns:

  1. The hypothesis
  2. The attention at each time step.



#### Undocumented methods ####

<a name="onmt.Beam:sort_best"></a>
 * `onmt.Beam:sort_best()`
<a name="onmt.Beam:get_best"></a>
 * `onmt.Beam:get_best()`
