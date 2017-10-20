# Start with CUDA Torch dependencies 2
FROM kaixhin/cuda-torch:8.0
MAINTAINER OpenNMT <http://opennmt.net/>

# Needed libs for luarocks--audio and zmq server
RUN sudo apt-get install -y libzmq-dev libfftw3-dev libsox-dev

# Needed for test
RUN luarocks install luacheck && \
    luarocks install luacov

RUN luarocks install tds && \
    luarocks install dkjson && \
    luarocks install lua-zmq ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include && \
    luarocks install sundown && \
    luarocks install cwrap && \
    luarocks install paths && \
    luarocks install torch && \
    luarocks install nn && \
    luarocks install dok && \
    luarocks install gnuplot && \
    luarocks install qtlua && \
    luarocks install qttorch && \
    luarocks install luafilesystem && \
    luarocks install penlight && \
    luarocks install sys && \
    luarocks install xlua && \
    luarocks install image && \
    luarocks install optim && \
    luarocks install lua-cjson && \
    luarocks install trepl && \
    luarocks install nnx && \
    luarocks install threads && \
    luarocks install graphicsmagick && \
    luarocks install argcheck && \
    luarocks install audio && \
    luarocks install signal && \
    luarocks install bit32

