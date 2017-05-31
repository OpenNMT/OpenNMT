OpenNMT supports ensemble decoding. It is a common technique to average the prediction of several models which usually yields better translation results. These models can be selected from the last few epochs, from trainings using different random seed or from multiple training histories.

!!! warning "Warning"
    Models must have the same target vocabulary.

To decode using multiple models, simply pass them as a list to the `-model` option:

```bash
th translate.lua -model model1.t7 model2.t7 model3.t7 -src src.txt
```

All other translation options are compatible with ensemble decoding.

## Multi GPU

As ensemble decoding requires lots of computational and memory resources, you usually want to use multiple GPUs. You can pass multiple GPU identifiers to the `-gpuid` option:

```bash
th translate.lua -model model1.t7 model2.t7 model3.t7 -src src.txt -gpuid 1 2 3
```

!!! note "Note"
    If less GPU devices than models are provided, the translation will cycle over provided devices.

Each models will be executed in its own thread. However, uncontrolled low-level synchronizations usually make ensemble decoding slower than single decoding.
