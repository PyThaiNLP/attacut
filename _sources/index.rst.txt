(beta) Fast and Reasonably Tokenizer for Thai
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:
        pages/*


.. figure:: figures/attacut-sych.png
    :align: center

    **TL;DR**: 3-Layer Dilated CNN on syllable and character features.

Usage
----

Installatation
^^^^
.. code-block:: bash

    pip install attacut

Command-Line Interface
^^^^
.. code-block:: bash

    $ attacut-cli -h
    AttaCut: Fast and Reasonably Accurate Tokenizer for Thai

    Usage:
    attacut-cli <src> [--dest=<dest>] [--model=<model>]
    attacut-cli (-h | --help)

    Options:
    -h --help         Show this screen.
    --model=<model>   Model to be used [default: attacut-sc].
    --dest=<dest>     If not specified, it'll be <src>-tokenized-by-<model>.txt


Higher-Level Interface
^^^^

.. code-block:: python

    from attacut import Tokenizer

    atta = Tokenizer() # default model: attacut-sc
    atta.tokenizer(txt)


AttaCut will be soon integrated into PyThaiNLP's ecosystem. Please see `PyThaiNLP #28 <https://github.com/PyThaiNLP/pythainlp/issues/258>`_ for recent updates.






Introduction
----

- What is AttaCut?
- reason why we developed this?
- when deepcut fails? [#deepcut]_

Evaluation
----

Quality
^^^^

Speed
^^^^


Development
----




References
----

.. [#deepcut] `Some jee <https://colab.research.google.com/drive/1Kb_Fhh6bS0sC2k3ovi2ce8AaWqFXNgIT>`_

.. [#aja] Ajavv somethinglink
