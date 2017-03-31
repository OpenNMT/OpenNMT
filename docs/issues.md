## luajit: out of memory

This most likely happened when training a model with long sequences and the LuaJIT memory limit was reached. You will need to switch to Lua 5.2 instead.

## THCudaCheck FAIL [...]: out of memory

This means your model was too large to fit on the available GPU memory.

To work around this error during training, follow these steps in order and stop when the training no more fails:

* Prefix your command line with `THC_CACHING_ALLOCATOR=0`
* Reduce the `-max_batch_size` value (64 by default)
* Reduce the `-src_seq_length` and `-tgt_seq_length` values during the preprocessing
* Reduce your model size (`-layers`, `-rnn_size`, etc.)

## unknown Torch class <torch.CudaTensor>

This means you wanted to load a GPU model but did not use the `-gpuid` option to define which GPU to use.
