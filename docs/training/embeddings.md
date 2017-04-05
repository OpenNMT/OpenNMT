Word embeddings are learned using a lookup table. Each word is assigned to a random vector within this table that is simply updated with the gradients coming from the network.

## Pretrained

When training with small amounts of data, performance can be improved by starting with pretrained embeddings. The arguments `-pre_word_vecs_dec` and `-pre_word_vecs_enc` can be used to specify these files.

The pretrained embeddings must be manually constructed Torch serialized tensors that correspond to the source and target dictionary files. For example:

```lua
local vocab_size = 50004
local embedding_size = 500

local embeddings = torch.Tensor(vocab_size, embedding_size):uniform()

torch.save('enc_embeddings.t7', embeddings)
```

where `embeddings[i]` is the embedding of the \(i\)-th word in the vocabulary.

## Fixed

By default these embeddings will be updated during training, but they can be held fixed using `-fix_word_vecs_enc` and `-fix_word_vecs_dec` options. These options can be enabled or disabled during a retraining.

## Extraction

The `tools/extract_embeddings.lua` can be used to extract the model word embeddings into text files. They can then be easily transformed into another format for visualization or processing.
