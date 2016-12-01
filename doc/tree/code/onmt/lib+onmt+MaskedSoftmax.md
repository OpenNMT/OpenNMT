<a name="onmt.MaskedSoftmax.dok"></a>


## onmt.MaskedSoftmax ##

 A batched-softmax wrapper to mask the probabilities of padding.

    AXXXAA
    AXXAAA
    AXXXXX



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/onmt/MaskedSoftmax.lua#L21">[src]</a>
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
