# Tokenization tests

## Adding new tests

1. Create the input raw text file `<name>_<mode>_<sep_annotate>_<case_feature>(_<bpe_model>).raw`, where:
   * `<name>` is the name of the test case without underscore
   * `<mode>` is the value of the `-mode` option on `tokenize.lua`
   * `<sep_annotate>` is the marker for setting the `-sep_annotate` option on `tokenize.lua`
   * `<case_feature>` is the value of the `-case_feature` option on `tokenize.lua` and `detokenize.lua`
   * (optional) `<bpe_model>` is the name of file (without .bpe suffix) for the `-bpe_model` option on `tokenize.lua`
2. Create the expected tokenized output file `<name>.tokenized`.
3. (optional) Create the expected detokenized output file `<name>.detokenized`.
   If this file is not provided, the detokenization of `<name>.tokenized` must match the raw input text.

## Running tests

1. Go to the top-level OpenNMT directory
2. `sh test/tokenization/test.sh`
