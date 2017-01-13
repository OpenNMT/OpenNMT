# Tools

This directory contains additional tools.

## Tokenization/Detokenization

### Dependencies

* `bit32` for Lua < 5.2

### Tokenization
To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-mode`: can be `aggressive` or `conservative` (default). In conservative mode, letters, numbers and '_' are kept in sequence, hyphens are accepted as part of tokens. Finally inner characters `[.,]` are also accepted (url, numbers).
* `-joiner_annotate`: if set, add joiner marker to indicate separator-less or BPE tokenization. The marker is defined by `-joiner` option and by default is joined to tokens (preference on symbol, then number, then letter) but can be independent if `-joiner_new` option is set.
* `-joiner`: default (￭) - the joiner marker
* `-joiner_new`: if set, the joiner is an independent token
* `-case_feature`: generate case feature - and convert all tokens to lowercase
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercased (opennmt)
  * `U`: token is uppercased (OPENNMT)
  * `C`: token is capitalized (Opennmt)
  * `M`: token case is mixed (OpenNMT)
* `-bpe_model`: Apply Byte Pair Encoding if the BPE model path is given
* `-nparallel`: Number of parallel thread to run the tokenization
* `-batchsize`: Size of each parallel batch - you should not change except if low memory

Note:

* `￨` is the feature separator symbol - if such character is used in source text, it is replace by its non presentation form `│`
* `￭` is the default joiner marker (generated in `-joiner_annotate marker` mode) - if such character is used in source text, it is replace by its non presentation form `■`

### Detokenization

If you activate `joiner_annotate` marker, the tokenization is reversible - just use:

```
th tools/detokenize.lua [-case_feature] [-joiner xx] < file.tok > file.detok
```

## Release model

After training a model on the GPU, you may want to release it to run on the CPU with the `release_model.lua` script.

```
th tools/release_model.lua -model model.t7 -gpuid 1
```

By default, it will create a `model_release.t7` file. See `th tools/release_model.lua -h` for advanced options.

## Translation Server

OpenNMT includes a translation server for running translate remotely. This also is an
easy way to use models from other languages such as Java and Python.

### Dependencies

* `lua-zmq`
* `json`

### Installation

The server uses the 0MQ for RPC. You can install 0MQ and the Lua bindings on Ubuntu by running:

```
sudo apt-get install libzmq-dev
luarocks install json
luarocks install lua-zmq ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include
```

Also you will need to install the OpenNMT as a library.

```
luarocks make rocks/opennmt-scm-1.rockspec
```

The translation server can be run using any of the arguments from `translate.lua`.

```
th tools/translation_server.lua -host ... -port ... -model ...
```

**Note:** the default host is set to `127.0.0.1` which only allows local access. If you want to support remote access, use `0.0.0.0` instead.

It runs as a message queue that takes in a JSON batch of src sentences. For example the following 5 lines of Python
code can be used to send a single sentence for translation.

```python
import zmq, sys, json
sock = zmq.Context().socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:5556")
sock.send(json.dumps([{"src": " ".join(sys.argv[1:])}]))
print sock.recv()
```

For a longer example, see our <a href="http://github.com/OpenNMT/Server/">Python/Flask server</a> in development.
