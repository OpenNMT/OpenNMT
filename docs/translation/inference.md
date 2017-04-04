## Release models

After training a model, you may want to release it for inference only by using the `release_model.lua` script. A released model takes less space on disk and is compatible with both CPU and GPU translation.

```bash
th tools/release_model.lua -model model.t7 -gpuid 1
```

By default, it will create a `model_release.t7` file. See `th tools/release_model.lua -h` for advanced options.

**Note:**

* a GPU is required to load non released models
* released models can no longer be used for training

## Inference engine

CTranslate is a C++ implementation of `translate.lua` for integration in existing products. Take a look at the [GitHub project](https://github.com/OpenNMT/CTranslate) for more information.
