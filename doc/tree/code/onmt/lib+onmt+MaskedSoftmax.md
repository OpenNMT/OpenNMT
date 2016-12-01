<a name="onmt.MaskedSoftmax.dok"></a>


## onmt.MaskedSoftmax ##

 A batched-softmax wrapper to mask the probabilities of padding.

    AXXXAA
    AXXAAA
    AXXXXX



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/MaskedSoftmax.lua#L21">[src]</a>
<a name="onmt.MaskedSoftmax"></a>


### onmt.MaskedSoftmax(source_sizes, source_length, beam_size) ###

 A nn-style module that applies a softmax on input that gives no weight to the left padding.

Parameters:

  * `source_sizes` -  the true lengths (with left padding).
  * `source_length` - the max length in the batch `beam_size`.
  * `beam_size` - beam size ${K}



#### Undocumented methods ####

<a name="onmt.MaskedSoftmax:updateOutput"></a>
 * `onmt.MaskedSoftmax:updateOutput(input)`
<a name="onmt.MaskedSoftmax:updateGradInput"></a>
 * `onmt.MaskedSoftmax:updateGradInput(input, gradOutput)`
<a name="onmt.MaskedSoftmax:accGradParameters"></a>
 * `onmt.MaskedSoftmax:accGradParameters(input, gradOutput, scale)`
