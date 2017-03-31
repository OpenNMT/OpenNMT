## Single GPU

The GPU is simply selected with the `-gpuid` option which takes as argument a 1-indexed identifier of the device to use.

## Multi GPU

OpenNMT supports *data parallelism* during the training. This technique allows the use of several GPUs by training batches in parallel on different *network replicas*. To enable this option, assign a list of comma-separated GPU identifier to the `-gpuid` option. For example:

```
th train.lua -data data/demo-train.t7 -save_model demo -gpuid 1,2,4
```

will use the first, the second and the fourth GPU of the machine.

There are 2 different modes:

* **synchronous parallelism** (default): in this mode, each replica processes in parallel a different batch at each iteration. The gradients from each replica are accumulated, and parameters updated and synchronized. Note that when using `N` GPU(s), the actual batch size is `N * max_batch_size`.
* **asynchronous parallelism** (`-async_parallel` flag): in this mode, the different replicas are independently
calculating their own gradient, updating a master copy of the parameters and getting updated values
of the parameters. Note that a GPU core is dedicated to storage of the master copy of the parameters and is not used for training. Also, to enable convergence at the beginning of the training, only one replica is working for the first `-async_parallel_minbatch` iterations.
