AttaCut: Fast and Reasonably Accurate Word Tokenizer for Thai
====================================================================


|travis_ic| |pypiversion_ic| |pypidownload_ic| |arxiv_ic| |license_ic| |github_ic|

.. toctree::
    :maxdepth: 2
    :hidden:

    NLP 101 <overview>
    survey
    benchmark
    training
    acknowledgement
    FAQs <faqs>
    misc


.. figure:: figures/attacut-sych.png
    :align: center

    **TL;DR**: 3-Layer Dilated CNN on syllable and character features.
    It's **6x faster** than DeepCut (SOTA) while its WL-f1 on BEST [#best]_ is **91%**, only 2% lower.

Installatation
--------------
.. code-block:: bash

    pip install attacut


**Note:** For **Windows** Users, please install **torch** before
running the command above. Visit `PyTorch.org <https://pytorch.org>`_ for
further instruction.

Usage
-----


Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    $ attacut-cli -h
    AttaCut: Fast and Reasonably Accurate Word Tokenizer for Thai

    Usage:
    attacut-cli <src> [--dest=<dest>] [--model=<model>]
    attacut-cli (-h | --help)

    Options:
    -h --help         Show this screen.
    --model=<model>   Model to be used [default: attacut-sc].
    --dest=<dest>     If not specified, it'll be <src>-tokenized-by-<model>.txt


High-Level API
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from attacut import tokenize, Tokenizer

    # tokenize `txt` using our best model `attacut-sc`
    words = tokenize(txt)

    # alternatively, an AttaCut tokenizer might be instantiated directly,
    # allowing one to specify whether to use attacut-sc or attacut-c.
    atta = Tokenizer(model="attacut-sc")
    words = atta.tokenize(txt)


AttaCut will be soon integrated into PyThaiNLP's ecosystem. Please see `PyThaiNLP #28 <https://github.com/PyThaiNLP/pythainlp/issues/258>`_ for recent updates


.. |travis_ic| image:: https://travis-ci.org/PyThaiNLP/attacut.svg?branch=master
    :target: https://travis-ci.org/PyThaiNLP/attacut
.. |pypiversion_ic| image:: https://img.shields.io/pypi/v/attacut
    :target: https://pypi.org/project/attacut/
.. |pypidownload_ic| image:: https://img.shields.io/pypi/dw/attacut
    :target: https://pypi.org/project/attacut/
.. |license_ic| image:: https://img.shields.io/pypi/l/attacut
.. |github_ic| image:: https://img.shields.io/github/stars/pythainlp/attacut?style=social
    :target: https://github.com/PyThaiNLP/attacut
.. |arxiv_ic| image:: http://img.shields.io/badge/arXiv-1911.07056-yellow.svg?style=flat
    :target: https://arxiv.org/abs/1911.07056

.. [#best] NECTEC. BEST: Benchmark for Enhancing the Standard of Thai language processing, 2010.
