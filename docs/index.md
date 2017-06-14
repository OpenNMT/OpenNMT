# OpenNMT project

OpenNMT project is composed of 3 main repositories:

* [OpenNMT lua](https://github.com/OpenNMT/OpenNMT) (_aka_ OpenNMT): the main project written using [torch](http://torch.ch) in lua. Optimized and stable code for production and large scale experiments.
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py): light version of OpenNMT in [pythorch](http://pytorch.org), initially created by FAIR team as a sample project for pytorch. This version, more easy to modify is more for research purpose does not include all features.
* [OpenNMT-C](https://github.com/OpenNMT/CTranslate) (_aka_ CTranslate): C optimized decoder for the models created with OpenNMT.

OpenNMT is a generic deep learning framework mainly specialized in sequence-to-sequence models covering a variety of tasks like [machine translation](/applications/#machine-translation) (NMT), [summarization](/applications/#summarization), but can also be used for non textual input, for instance [im2text](/applications/#im2text) is recognizing latex formulas in images, or [speech recognition](/applications/#speech-recognition). The framework has also been extended for other non sequence-to-sequence tasks like [language modelling](/applications/#language-modelling), and [sequence taggers](/applications/#sequence-tagging).

All these applications are reusing and sometime extending a collection of generic easy-to-reused modules, [encoders](/training/models/#encoders), [decoders](/training/models/#decoders), [embeddings](/training/embeddings/), [attention models](/training/models/#attention-model) and more...

The framework is implemented to be as generic as possible and can be used either through commandline applications, client-server, or librairies.

The project is self-contained and ready to use either for research and for production.

OpenNMT project is an open-source initiative derivated from [seq2seq-attn](https://github.com/SYSTRAN/seq2seq-attn), initially created by Kim Yoon at HarvardNLP group.

## Additional resources

You can find additional help or tutorials in the following resources:

* [Forum](http://forum.opennmt.net/)
* [Gitter channel](https://gitter.im/OpenNMT/openmt)

!!! note "Note"
    If you find an error in this documentation, please consider [opening an issue](https://github.com/OpenNMT/OpenNMT/issues/new) or directly submitting a modification by clicking on the edit button at the top of a page.
