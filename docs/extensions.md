OpenNMT is explicitly separated out into a library and application section. All modeling and training code can be directly used within other Torch applications.

##  Image-to-Text

As an example use case we have released an extension for translating from images-to-text. This model replaces the source-side word embeddings with a convolutional image network. The full model is
available at <a href="https://github.com/opennmt/im2text">OpenNMT/im2text</a>.
