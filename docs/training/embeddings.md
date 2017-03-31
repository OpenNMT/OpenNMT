## Pre-trained

When training with small amounts of data, performance can be improved by starting with pretrained embeddings. The arguments `-pre_word_vecs_dec` and `-pre_word_vecs_enc` can be used to specify these files.

The pretrained embeddings must be manually constructed Torch serialized tensors that correspond to the source and target dictionary files. For example:

```lua
local emb = torch.Tensor(vocab_size, embedding_size):uniform()
torch.save('emb.t7', emb)
```

By default these embeddings will be updated during training, but they can be held fixed using `-fix_word_vecs_enc` and `-fix_word_vecs_dec` options.
