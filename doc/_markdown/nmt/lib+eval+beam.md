<a name="nmt.Beam.dok"></a>


## nmt.Beam ##

 Class for managing the beam search process. 

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/eval/beam.lua#L26">[src]</a>
<a name="nmt.Beam"></a>


### nmt.Beam(size) ###

Constructor

Parameters:

  * `size` : The beam `K`.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/eval/beam.lua#L47">[src]</a>
<a name="nmt.Beam:get_current_state"></a>


### nmt.Beam:get_current_state() ###

 Get the outputs for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/eval/beam.lua#L52">[src]</a>
<a name="nmt.Beam:get_current_origin"></a>


### nmt.Beam:get_current_origin() ###

 Get the backpointers for the current timestep.

<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/eval/beam.lua#L66">[src]</a>
<a name="nmt.Beam:advance"></a>


### nmt.Beam:advance(out, attn_out) ###

 Given prob over words for every last beam `out` and attention
 `attn_out`. Compute and update the beam search.

Parameters:

  * `out`- probs at the last step
  * `attn_out`- attention at the last step

Returns: true if beam search is complete.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/84822a44954b241391f9198ab595f845feb7a6b0/lib/eval/beam.lua#L132">[src]</a>
<a name="nmt.Beam:get_hyp"></a>


### nmt.Beam:get_hyp(k) ###

 Walk back to construct the full hypothesis `k`.

Parameters:

  * `k` - the position in the beam to construct.

Returns:

  1. The hypothesis
  2. The attention at each time step.



#### Undocumented methods ####

<a name="nmt.Beam:sort_best"></a>
 * `nmt.Beam:sort_best()`
<a name="nmt.Beam:get_best"></a>
 * `nmt.Beam:get_best()`
