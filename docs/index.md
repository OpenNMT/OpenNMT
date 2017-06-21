# OpenNMT project

OpenNMT project is composed of 3 main repositories:

* [OpenNMT-Lua](https://github.com/OpenNMT/OpenNMT) (a.k.a. OpenNMT): the main project developed with [LuaTorch](http://torch.ch).<br/>Optimized and stable code for production and large scale experiments.
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py): light version of OpenNMT using [PyTorch](http://pytorch.org).<br/>Initially created by the Facebook AI research team as a sample project for PyTorch, this version is easier to extend and is suited for research purpose but does not include all features.
* [OpenNMT-C](https://github.com/OpenNMT/CTranslate) (a.k.a. CTranslate): C++ inference engine for OpenNMT models.

OpenNMT is a generic deep learning framework mainly specialized in sequence-to-sequence models covering a variety of tasks such as [machine translation](/applications/#machine-translation), [summarization](/applications/#summarization), [image to text](/applications/#image-to-text), and [speech recognition](/applications/#speech-recognition). The framework has also been extended for other non sequence-to-sequence tasks like [language modelling](/applications/#language-modelling) and [sequence tagging](/applications/#sequence-tagging).

All these applications are reusing and sometimes extending a collection of easy-to-reuse modules: [encoders](/training/models/#encoders), [decoders](/training/models/#decoders), [embeddings layers](/training/embeddings/), [attention layers](/training/models/#attention-model), and more.

The framework is implemented to be as generic as possible and can be used either via command line applications, client-server, or libraries.

The project is self-contained and ready to use for both research and production.

*OpenNMT project is an open-source initiative derivated from [seq2seq-attn](https://github.com/SYSTRAN/seq2seq-attn), initially created by Kim Yoon at HarvardNLP group.*

## Additional resources

You can find additional help or tutorials in the following resources:

* [Forum](http://forum.opennmt.net/)
* [Gitter channel](https://gitter.im/OpenNMT/openmt)

!!! note "Note"
    If you find an error in this documentation, please consider [opening an issue](https://github.com/OpenNMT/OpenNMT/issues/new) or directly submitting a modification by clicking on the edit button at the top of a page.
