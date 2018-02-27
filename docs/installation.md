## Standard

1\. [Install Torch](http://torch.ch/docs/getting-started.html)

2\. Install additional dependencies:

```bash
luarocks install tds
luarocks install bit32 # if using LuaJIT
```

3\. Clone the OpenNMT repository:

```bash
git clone https://github.com/OpenNMT/OpenNMT
cd OpenNMT
```

And you are ready to go! Take a look at the [quickstart](quickstart.md) to familiarize yourself with the main training workflow.

## Docker

1\. [Install `nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) if using GPUs

2\. Pull and run the latest OpenNMT image:

```bash
sudo nvidia-docker run -it opennmt/opennmt:latest
```

## Amazon EC2

The best way to do this is through Docker. We have a public AMI with the preliminary CUDA drivers installed: `ami-c12f86a1`. Start a P2/G2 GPU instance with this AMI and run the `nvidia-docker` command above to get started.
