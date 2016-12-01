<a name="onmt.WordEmbedding.dok"></a>


## onmt.WordEmbedding ##

 nn unit. Maps from word ids to embeddings. Slim wrapper around
nn.LookupTable to allow fixed and pretrained embeddings.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/onmt/WordEmbedding.lua#L16">[src]</a>
<a name="onmt.WordEmbedding"></a>


### onmt.WordEmbedding(vocab_size, vec_size, pre_trained, fix) ###


Parameters:

  * `vocab_size` - size of the vocabulary
  * `vec_size` - size of the embedding
  * `pre_trainined` - path to a pretrained vector file
  * `fix` - keep the weights of the embeddings fixed.



#### Undocumented methods ####

<a name="onmt.WordEmbedding:updateOutput"></a>
 * `onmt.WordEmbedding:updateOutput(input)`
<a name="onmt.WordEmbedding:updateGradInput"></a>
 * `onmt.WordEmbedding:updateGradInput(input, gradOutput)`
<a name="onmt.WordEmbedding:accGradParameters"></a>
 * `onmt.WordEmbedding:accGradParameters(input, gradOutput, scale)`
