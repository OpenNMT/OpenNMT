FROM nvidia/cuda:8.0-devel-ubuntu16.04 as torch_builder
ARG CUDA_ARCH
ARG ONMT_URL
ARG ONMT_REF

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
        unzip

# Compile Torch and OpenNMT dependencies.
ENV CUDA_ARCH=${CUDA_ARCH:-Common}
RUN git clone https://github.com/torch/distro.git /root/torch-distro --recursive && \
    cd /root/torch-distro && \
    mkdir /root/torch && \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    PREFIX=/root/torch ./install.sh
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

# Fetch OpenNMT.
ENV ONMT_URL=${ONMT_URL:-https://github.com/OpenNMT/OpenNMT.git}
ENV ONMT_REF=${ONMT_REF:-master}
RUN git clone --depth 1 --branch ${ONMT_REF} --single-branch ${ONMT_URL} /root/opennmt


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
ENV ONMT_DIR=/root/opennmt

COPY --from=torch_builder /root/torch ${TORCH_DIR}
COPY --from=torch_builder /root/sentencepiece ${SENTENCEPIECE_DIR}
COPY --from=torch_builder /root/opennmt ${ONMT_DIR}

ENV LUA_PATH="${TORCH_DIR}/share/lua/5.1/?.lua;${TORCH_DIR}/share/lua/5.1/?/init.lua;./?.lua"
ENV LUA_CPATH="${TORCH_DIR}/lib/lua/5.1/?.so;./?.so;${TORCH_DIR}/lib/?.so"
ENV PATH=${TORCH_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TORCH_DIR}/lib:${LD_LIBRARY_PATH}
ENV THC_CACHING_ALLOCATOR=0

WORKDIR $ONMT_DIR
