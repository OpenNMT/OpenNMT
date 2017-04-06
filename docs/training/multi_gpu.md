OpenNMT can make use of multiple GPU during the training by implementing **data parallelism**. This technique trains batches in parallel on different network replicas. To use data parellelism, assign a list of comma-separated GPU identifiers to the `-gpuid` option. For example:

```bash
th train.lua -data data/demo-train.t7 -save_model demo -gpuid 1,2,4
```

will use the first, the second and the fourth GPU of the machine as returned by the CUDA API.

!!! note "Note"
    `nvidia-smi` enumerates devices based on the driver API which can be in a different order than the CUDA API.

## Synchronous

In this default mode, each replica processes in parallel a different batch at each iteration. The gradients from each replica are accumulated, and parameters updated and synchronized.

!!! warning "Warning"
    When using \(N\) GPU(s), the actual batch size is \(N \times\) `-max_batch_size`.

## Asynchronous

*(Also known as asynchronous SGD or downpour SGD.)*

In this mode enabled with the `-async_parallel` flag, the different replicas are independently
calculating their own gradients, updating a master copy of the parameters and getting updated values
of the parameters. To enable convergence at the beginning of the training, only one replica is working for the first `-async_parallel_minbatch` iterations to prepare a better initialization for the asynchronous part.

!!! warning "Warning"
    A GPU core is dedicated to store the master copy of the parameters and is not used for training.

!!! note "Note"
    As training logs and saving require synchronization, consider using higher `-report_every` and `-save_every` values.
