# Style Guide

## Comments

* Comments should follow:
https://github.com/deepmind/torch-dokx/blob/master/doc/usage.md

* All non-private method should have dokx comments describing input/output.

* All classes should have a class docstring at the top of the file.

* All comments should be on their own line, and be a complete English
sentence with capitalization.


* Comments are of the form:

```
--[[ Set target input directly.

Parameters:
  * `targetInput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Co
--]]
```

* Use torch-dokx and this command to build docs

```
dokx-build-package-docs -o docs .
google-chrome doc/index.html
```

* Use mkdocs to display the pretty version

```
pip install mkdocs
pip install python-markdown-math
cd doc/
bash mkdocs.sh
mkdoc serve
```

## Style

* Please run and correct all warnings from `luacheck` before sending a pull request.

```
luacheck *lua onmt
```

* All indentation should be 2 spaces.

* All variables, functions and methods should use camelCase.

* Do not use C-like extraneous parentheses or use less ';' instruction-separators

* Use space around operators as much as possible to make code readable

* add comments and extra-lines
