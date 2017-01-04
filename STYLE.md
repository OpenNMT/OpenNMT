# Style Guide

## Comments

* All non-private methods should have a [torch-dokx](https://github.com/deepmind/torch-dokx/blob/master/doc/usage.md) documentation describing input/output.

* All classes should have a class docstring at the top of the file.

* All comments should be on their own line, and be a complete English sentence with capitalization.

The torch-dokx documentation is automatically deployed [here](http://opennmt.net/OpenNMT/).

## Code

* All files should pass `luacheck`.

* Use object-oriented programming whenever appropriate.

* Avoid using global variables.

* Avoid writing functions with more than 100 lines.

* Use `onmt.utils.Logger` to log messages for the user.

* If a new module relies on command line options, consider defining a static `declareOpts(cmd)` function. See `onmt/translate/Translator.lua` for an example.

## Style

* All indentation should be 2 spaces.

* All variables, functions and methods should use camelCase.

* Commande line options and scripts name should use snake_case.

* Use spaces around operators to increase readability.
