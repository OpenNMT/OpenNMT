<a name="nmt.BiEncoder.dok"></a>


## nmt.BiEncoder ##


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/f651283e010895d259d9defa2c8fba8ce80e74f3/lib/biencoder.lua#L46">[src]</a>
<a name="nmt.BiEncoder"></a>


### nmt.BiEncoder(args, merge, net_fwd, net_bwd) ###

 Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).



#### Undocumented methods ####

<a name="nmt.BiEncoder:resize_proto"></a>
 * `nmt.BiEncoder:resize_proto(batch_size)`
<a name="nmt.BiEncoder:forward"></a>
 * `nmt.BiEncoder:forward(batch)`
<a name="nmt.BiEncoder:backward"></a>
 * `nmt.BiEncoder:backward(batch, grad_states_output, grad_context_output)`
<a name="nmt.BiEncoder:training"></a>
 * `nmt.BiEncoder:training()`
<a name="nmt.BiEncoder:evaluate"></a>
 * `nmt.BiEncoder:evaluate()`
<a name="nmt.BiEncoder:convert"></a>
 * `nmt.BiEncoder:convert(f)`
