# Re-compile git with openssl rather than gnutls because of issue with AWS CodeCommit
# Using multi-stage builds docker feature
FROM ubuntu:14.04 as builder
RUN sudo sed -i 's/# \(deb-src.*\)/\1/g' /etc/apt/sources.list
RUN sudo apt-get update && \
    sudo apt-get install -y build-essential fakeroot dpkg-dev && \
    sudo apt-get build-dep -y git && \
    sudo apt-get install -y libcurl4-openssl-dev
RUN mkdir /tmp/git-openssl && \
    cd /tmp/git-openssl && \
    apt-get source git && \
    dpkg-source -x git_1.9.1-1ubuntu0.6.dsc && \
    cd git-1.9.1 && \
    sed -i 's/libcurl4-gnutls-dev/libcurl4-openssl-dev/g' debian/control && \
    sed -i '/^TEST =test$/d' debian/rules && \
    sudo dpkg-buildpackage -rfakeroot -b

# Start with CUDA Torch dependencies 2
FROM kaixhin/cuda-torch:8.0
MAINTAINER OpenNMT <http://opennmt.net/>

#use git-openssl from previous build stage
COPY --from=builder /tmp/git-openssl/git_1.9.1-1ubuntu0.6_amd64.deb /tmp/
RUN sudo dpkg -i /tmp/git_1.9.1-1ubuntu0.6_amd64.deb

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

