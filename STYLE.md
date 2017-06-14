# Style Guide

## Comments

* When documenting an interface, use [torch-dokx](https://github.com/deepmind/torch-dokx/blob/master/doc/usage.md) documentation style.

* All comments should be on their own line, and be a complete English sentence with capitalization.

## Design

* All files should pass `luacheck`.

* Use object-oriented programming whenever appropriate.

* Avoid using global variables.

* Avoid writing functions with more than 100 lines.

* Use `onmt.utils.Logger` to log messages for the user.

* If a new module relies on command line options, consider defining a static `declareOpts(cmd)` function. See `onmt/translate/Translator.lua` for an example.

## Formatting

* All indentation should be 2 spaces.

* All variables, functions and methods should use camelCase.

* Commande line options and scripts name should use snake_case.

* Use spaces around operators to increase readability.

* Avoid lines with more than 100 columns.
