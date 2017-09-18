# Start with CUDA Torch dependencies 2
FROM kaixhin/cuda-torch:8.0
MAINTAINER OpenNMT <http://opennmt.net/>

RUN sudo apt-get install -y libzmq-dev

# Needed libs for luarocks--audio
RUN sudo apt-get install -y libfftw3-dev libsox-dev

RUN luarocks install tds
RUN luarocks install dkjson
RUN luarocks install lua-zmq ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include
RUN luarocks install sundown
RUN luarocks install cwrap
RUN luarocks install paths
RUN luarocks install torch
RUN luarocks install nn
RUN luarocks install dok
RUN luarocks install gnuplot
RUN luarocks install qtlua
RUN luarocks install qttorch
RUN luarocks install luafilesystem
RUN luarocks install penlight
RUN luarocks install sys
RUN luarocks install xlua
RUN luarocks install image
RUN luarocks install optim
RUN luarocks install lua-cjson
RUN luarocks install trepl
RUN luarocks install nnx
RUN luarocks install threads
RUN luarocks install graphicsmagick
RUN luarocks install argcheck
RUN luarocks install audio
RUN luarocks install signal
RUN luarocks install bit32

#Needed for test
RUN luarocks install luacheck && \
    luarocks install luacov
