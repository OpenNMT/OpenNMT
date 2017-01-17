# Start with CUDA Torch dependencies 2
FROM kaixhin/cuda-torch-deps:2-8.0
MAINTAINER Kai Arulkumaran <design@kaixhin.com>

# Restore Torch7 installation script
RUN sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

# Install CUDA libraries
RUN luarocks install cutorch && \
  luarocks install cunn && \
  luarocks install cudnn && \
  luarocks install tds
