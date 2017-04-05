OpenNMT also provides some tools and options to extract model information and input encoding.

## Word embeddings

The `tools/extract_embeddings.lua` can be used to extract the model word embeddings into text files. They can then be easily transformed into another format for visualization or processing.

## Sentence encoding

The translation script `translate.lua` supports an optional flag `-dump_input_encoding` to output sentence encoding instead of translation. The encoding of a sentence is the last hidden state of the encoder.
