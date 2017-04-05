By default, OpenNMT's scripts can only be called from the root of OpenNMT's directory. If calling the scripts from any directory is more convenient to you, you need to extend the `LUA_PATH`:

```bash
export LUA_PATH="$LUA_PATH;/path/to/OpenNMT/?.lua"
```

## Configuration files

You can pass options using a configuration file. The file has a simple key-value syntax with one `option = value` per line. Here is an example:

```text
$ cat generic.txt
rnn_size = 600
layers = 4
brnn = true
save_model = generic
```

It handles empty lines and ignores lines prefixed with `#`.

You can then pass this file along other options on the command line:

```bash
th train.lua -config generic.txt -data data/demo-train.t7 -gpuid 1
```

If an option appears both in the file and on the command line, the file takes priority.

## Boolean flags

Flags are options that do not take arguments. For example the option `-brnn` enables bidirectional encoder when added to the command line.

However, flags that are enabled by default can take `0` as argument to disable them. For example, input feeding is disabled with `-input_feed 0`.

## Multiple arguments

Some options can take multiple arguments. Unless otherwise noted, they accept a list of comma-separated values without spaces:

```text
-option_name value1,value2,value3
```
