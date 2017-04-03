Word embeddings are learned using a lookup table. Each word is assigned to a random vector within this table that is simply updated with the gradients coming from the network.

## Pretrained

When training with small amounts of data, performance can be improved by starting with pretrained embeddings. The arguments `-pre_word_vecs_dec` and `-pre_word_vecs_enc` can be used to specify these files.

The pretrained embeddings must be manually constructed Torch serialized tensors that correspond to the source and target dictionary files. For example:

```lua
local vocab_size = 50004
local embedding_size = 500

torch.save('emb.t7', torch.Tensor(vocab_size, embedding_size):uniform())
```

## Fixed

By default these embeddings will be updated during training, but they can be held fixed using `-fix_word_vecs_enc` and `-fix_word_vecs_dec` options. These options can be enabled or disabled during a retraining.
