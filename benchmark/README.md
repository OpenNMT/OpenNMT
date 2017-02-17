# Prerequisites

1. Have a GitHub account
2. Login and get your API key at http://scorer.nmt-benchmark.net/api

# Submit a model

## Generate the model metadata

```lua
th benchmark/generate_metadata.lua -model ende_model.t7 -save_data ende_model.json -name ende-baseline-20170206 -language_pair ende
```

* If `ende_model.t7` is a GPU model, add `-gpuid 1` to the command line.
* See `th benchmark/generate_metadata.lua -h` for additional options to describe your model and training (features, tokenization, etc.)

## Submit the model

```python
python benchmark/submitSystem.py --apikey <YOUR API KEY> ende_model.json
```

## Get test files for a given language pair

```python
python benchmark/getTests.py --apikey <YOUR API KEY> --src <SOURCE LANGUAGE> --tgt <TARGET LANGUAGE> [--basedir .]
```

generates a directory named `benchmark-nmt-test-SRCTGT-YYYY-MM-DD` in `basedir` with all the test sets available for that language pair.

## Submit translation output and get scores

```python
python benchmark/submitOutput.py --systemId <SYSTEM ID> --apikey <YOUR API KEY> --info FILE.info --output FILE.out
```
