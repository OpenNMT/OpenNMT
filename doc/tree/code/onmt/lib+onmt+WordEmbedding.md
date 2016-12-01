<a name="onmt.WordEmbedding.dok"></a>


## onmt.WordEmbedding ##

 nn unit. Maps from word ids to embeddings. Slim wrapper around
nn.LookupTable to allow fixed and pretrained embeddings.


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/ecd46c8eee34474c91ab3606f3e19a1b9db13b22/lib/onmt/WordEmbedding.lua#L16">[src]</a>
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
