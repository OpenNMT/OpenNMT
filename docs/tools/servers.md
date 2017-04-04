OpenNMT provides simple translation servers to easily showcase your results remotely.

### REST

You can use an easy REST syntax to simply send plain text. Sentences will be tokenized, translated and then detokenized using OpenNMT tools.

The server uses the `restserver-xavante` dependency, you need to install it by running:

```bash
luarocks install restserver-xavante
```

The translation server can be run using any of the arguments from `tokenize.lua` or `translate.lua`.

```bash
th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1 -host ... -port -case_feature -bpe_model ...
```

**Note:** the default host is set to `127.0.0.1` and default port to `7784`.

You can test it with a `curl` command locally or from any other client:

```bash
curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "Hello World }]' http://IP_address:7784/translator/translate
```

Answer will be embeeded in a JSON format, translated sentence in the `tgt` section. Additionnally you can get the attention matrix with the `-withAttn` option in the server command line.

## ZeroMQ

The server uses the 0MQ for RPC. You can install 0MQ and the Lua bindings on Ubuntu by running:

```bash
sudo apt-get install libzmq-dev
luarocks install dkjson
luarocks install lua-zmq ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include
```

The translation server can be run using any of the arguments from `translate.lua`.

```bash
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
