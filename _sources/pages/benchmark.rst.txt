.. _sec-benchmark:

Benchmarking
------------

We value reproducibility. Our experiments should be reproducible and expected
to have similar results when one tries. Therefore, we 1) desrcibe our
benchmarking procedure as complete as possible and 2) publish all the code with
fair amount of documentation.

If there is any doubt or unclear part,
please let us know. We are happy to clarify and improve the document.

Please note here that AttaCut models are denoted as **AttaCut-SC** and
**AttaCut-C**. The former is AttaCut with syllable and character features,
while the latter uses only character feature.


Tokenization Quality
^^^^^^^^^^^^^^^^^^^^
Tokenization quality is measured in terms of **precision**, **recall**, and
**f1**. We do the measurements in two levels, namely character and word.
Figure below describes how these metrics are computed:


.. figure:: ../figures/evaluation-long.png

    Character- and Word-Level Metrics for Word Tokenization


.. code-block::

    Character-Level:
    [P]recision = TP / ( TP + FP )
    [R]ecall = TP / ( TP + FN )
    f1 = 2PR / (P+R)

    Word-Level:
    P = #✓ / #◼︎ in prediction
    R = #✓ / #◼︎ in text

To increase reproducibility and ease further research, we have developed an
evaluation framework for this process. The framework contains two main
ingredients:

1. | **Bechmark CLI**
   | At the moment, this CLI can be found at `@pythainlp's tokenization-benchmark <https://github.com/PyThaiNLP/tokenization-benchmark>`_, but it will be soon released in the main PyThaiNLP package (version 2.1). Please see it this milestone [#milestone]_ for recent updates.
2. | **Result Visualization and Comparison Website**
   | This website serves as a tool for error analysis on tokenization results as well as a benchmark collection of other publicly available tokenizers.


    .. figure:: https://camo.githubusercontent.com/85984f46bb0db3e2bb86b16969b570b7faf4535a/68747470733a2f2f692e696d6775722e636f6d2f56564159485a4d2e706e67

        Tokenization Benchmark Visualization [#viz]_

Results
"""""""
...

Speed
^^^^^

Our speed benchmarking is done on standardized environments, namely Google Colab
and AWS's EC2 instances (t2.small & t2.medium).


Benchmarking on Google Colab
""""""""""""""""""""""""""""

Due to Google Colab's accessibilty and convenience, we use Google Colab for our
early speed benchmarking. In this experiment, we vary the length of input text
and measure the speed of tokenizers.

.. figure:: ../figures/colab-speed-benchmark.png

    Tokenization Time of PyThaiNLP's newmm, DeepCut, and AttaCut on Google Colab

From the figure above, we can see that our AttaCut models are significantly
faster than DeepCut.


Benchmarking on EC2 Instances
"""""""""""""""""""""""""""""

Practically, tokenization is part of NLP pipelines that is usually done on
cloud instances, such as AWS EC2, due to scalibility and cost efficiency.
Typically, these instances contain a couple of CPU cores and memory,
posing another challenge to services, i.e. tokenization, executued there.


Evaluating tokenizer's speed on such an instance allows us to get realistic
results and yet reproducible. We use the training set of Wisesight Sentiment
Corpus [#wisesight]_ as a input dataset. The corpus contains texts from social
media and online forum platforms. The training set has around 24,000 lines and
about 1.5M characters.

.. realistic setting, low resource device.. 

.. figure:: ../figures/speed-benchmark-ec2.png

    Wisesight's Training Set Tokenization Time of PyThaiNLP's newmm, DeepCut, and AttaCut on AWS Instances.

From the figure above, AttaCut models are fasters than other existing ML-based
tokenizers. More precisely, **AttaCut-SC** (our best model) is aroud **6x**
faster than **DeepCut**, the current state of the art word tokenizer for Thai,
while having a similar level of tokenization quality.


.. [#milestone] `PyThaiNLP 2.1 Milestone <https://github.com/PyThaiNLP/pythainlp/milestone/11>`_
.. [#viz] `Tokenization Benchmark Visualization <https://pythainlp.github.io/tokenization-benchmark-visualization/>`_
.. [#wisesight] `PyThaiNLP. Wisesight-Sentiment Corpus, 2019 <https://github.com/PyThaiNLP/wisesight-sentiment>`_