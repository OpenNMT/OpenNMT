FROM nvidia/cuda:8.0-devel-ubuntu16.04 as torch_builder

RUN apt-get update && \
    apt-get install -y \
        autoconf \
        automake \
        build-essential \
        cmake \
        curl \
        g++ \
        gcc \
        git \
        libprotobuf-dev \
        libprotobuf9v5 \
        libreadline-dev \
        libtool \
        libzmq-dev \
        pkg-config \
        protobuf-compiler \
        unzip \
        wget

# Fetch Intel MKL.
RUN mkdir /root/mkl && \
    wget https://anaconda.org/intel/mkl/2018.0.1/download/linux-64/mkl-2018.0.1-intel_4.tar.bz2 && \
    tar -xf mkl-2018.0.1-intel_4.tar.bz2 -C /root/mkl && \
    rm mkl-2018.0.1-intel_4.tar.bz2
ENV MKL_ROOT=/root/mkl
RUN rm -f $MKL_ROOT/lib/*vml* \
          $MKL_ROOT/lib/*ilp64* \
          $MKL_ROOT/lib/*blacs* \
          $MKL_ROOT/lib/*scalapack* \
          $MKL_ROOT/lib/*cdft* \
          $MKL_ROOT/lib/libmkl_tbb_thread.so \
          $MKL_ROOT/lib/libmkl_ao_worker.so

# Compile Torch and OpenNMT dependencies.
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH:-Common}
RUN git clone https://github.com/torch/distro.git /root/torch-distro --recursive && \
    cd /root/torch-distro && \
    mkdir /root/torch && \
    CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$MKL_ROOT/lib \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    PREFIX=/root/torch ./install.sh
RUN cp -r $MKL_ROOT/lib/* /root/torch/lib
RUN /root/torch/bin/luarocks install tds && \
    /root/torch/bin/luarocks install dkjson && \
    /root/torch/bin/luarocks install wsapi && \
    /root/torch/bin/luarocks install yaml && \
    /root/torch/bin/luarocks install bit32 && \
    /root/torch/bin/luarocks install luacheck && \
    /root/torch/bin/luarocks install luacov && \
    /root/torch/bin/luarocks install lua-zmq \
        ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include

# Install lua-sentencepiece
RUN git clone https://github.com/google/sentencepiece.git /root/sentencepiece-git && \
    cd /root/sentencepiece-git && \
    ./autogen.sh && \
    ./configure --prefix=/root/sentencepiece && \
    make && \
    make install && \
    cd /root && \
    rm -r /root/sentencepiece-git
RUN git clone https://github.com/OpenNMT/lua-sentencepiece.git /root/lua-sentencepiece && \
    cd /root/lua-sentencepiece && \
    CMAKE_LIBRARY_PATH=/root/sentencepiece/lib CMAKE_INCLUDE_PATH=/root/sentencepiece/include \
    /root/torch/bin/luarocks make lua-sentencepiece-scm-1.rockspec \
        LIBSENTENCEPIECE_DIR=/root/sentencepiece && \
    cd /root && \
    rm -r /root/lua-sentencepiece


FROM nvidia/cuda:8.0-runtime-ubuntu16.04
MAINTAINER OpenNMT <http://opennmt.net/>

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libprotobuf9v5 \
        libzmq1 \
        python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TORCH_DIR=/root/torch
ENV SENTENCEPIECE_DIR=/root/sentencepiece

COPY --from=torch_builder /root/torch ${TORCH_DIR}
COPY --from=torch_builder /root/sentencepiece ${SENTENCEPIECE_DIR}

ENV LUA_PATH="${TORCH_DIR}/share/lua/5.1/?.lua;${TORCH_DIR}/share/lua/5.1/?/init.lua;./?.lua"
ENV LUA_CPATH="${TORCH_DIR}/lib/lua/5.1/?.so;./?.so;${TORCH_DIR}/lib/?.so"
ENV PATH=${TORCH_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TORCH_DIR}/lib:${LD_LIBRARY_PATH}
ENV THC_CACHING_ALLOCATOR=0

ENV ONMT_DIR=/root/opennmt
COPY . $ONMT_DIR

WORKDIR $ONMT_DIR
