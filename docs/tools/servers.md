OpenNMT provides simple translation servers to easily showcase your results remotely.

### REST

Please use ```export THC_CACHING_ALLOCATOR=0``` to save memory on server side.

You can use an easy REST syntax to simply send plain text. Sentences will be tokenized, translated and then detokenized using OpenNMT tools.

The server uses the `restserver-xavante` dependency, you need to install it by running:

```bash
luarocks install restserver-xavante
```

The translation server can be run using any of the arguments from `tokenize.lua` or `translate.lua`.

Single Model Rest server (first version, kept for backward compatiblity)

```bash
th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1 -host ... -port -case_feature -bpe_model ...
```

!!! note "Note"
    The default host is set to `127.0.0.1` and default port to `7784`.

You can test it with a `curl` command locally or from any other client:

```bash
curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "Hello World" }]' http://IP_address:7784/translator/translate
```

Answer will be embedded in a JSON format, translated sentence in the `tgt` section. Additionally you can get the attention matrix with the `-withAttn` option in the server command line.

#### Multi Model Rest server

```bash
luarocks install yaml
```

This version supports multi models listed in a YAML config file.

Here is an example with two models:

```yaml
-
  model: '/NMTModels/en-fr/model-enfr_epoch600_3.03.t7'
  replace_unk: true
  mode: aggressive
  joiner_annotate: true
  case_feature: true
  segment_case: true
  beam_size: 5

-
  model: '/NMTModels/en-it/model-enit_epoch600_4.17.t7'
  replace_unk: true
  mode: aggressive
  joiner_annotate: true
  case_feature: true
  segment_case: true
  beam_size: 5
```

By default, it uses the file in `test/data/rest_config.yml` but you can modify with `--mode_config the location`.


```bash
th tools/rest_multi_models.lua -gpuid 1
```

!!! note "Note"
    The default host is set to `127.0.0.1` which only allows local access. If you want to support remote access, use `0.0.0.0` instead. Default port is 7784. You can change the unload time with `--unload_time Xsec`

You can test it with a `curl` command locally or from any other client:

You need to select the model id in the order of the config file.

```bash
curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "Hello World" , "id" : 1 }]' http://IP_address:7784/translator/translate
```


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

!!! note "Note"
    The default host is set to `127.0.0.1` which only allows local access. If you want to support remote access, use `0.0.0.0` instead.

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
