Word Tokenization for Thai
--------------------------

Research in word tokenization for Thai started around 1990. Over these 20 years,
there have been sevaral algorithms being prosed to address the problem. These algorithms
can be clustered into two categories, namely

1. | **Dictionary-based:**
   | Algorithms in this category rely on the use of dictionaries with a mechanism to decide whether to tokenize a particular sequence of characters. Some of algorithms are Chrome's v8BreakIterator [#icu]_  and PyThaiNLP's newmm [#newmm]_.

2. | **Learning-based:**
   | Unlike dictionary-based, algorithms in this group learn to split words based on labelled data. The learning problem is typically formulated as **binary classification** on sequence of characters.

   .. figure:: ../figures/binary-classification.png
        :width: 300px
        :align: center

        Binary Classification for Word Tokenization. **B** denotes a starting-word character, while **I** represents the opposite.

   | With the rise of neural networks, recent developments of Thai tokenizers are either Convolutional Neural Networks (CNNs) (i.e. DeepCut [#deepcut]_) or Recurrent Neural Networks (RNNs) (i.e. [#multicut]_, [#cantok]_, Sertis' Bi-GRUs [#sertis]_).

Generally, these categories have different advantages and disadvantages.
Dictionary-based algorithms are typically fast but with less capable when encountering unknown words.
On the other hand, learning-based approaches are usually qualitatively better and more adaptable to data from different domains; however, their computation is relatively slower.
Figure below summarizes current solutions into two axes: **Quality (Word-Level f1)** and **Inference time**.

   .. figure:: ../figures/previous-work-spectrum.png
        :align: center

        Quality and Inference Time of Existing Thai Word Tokenizers. Please see :ref:`sec-benchmark` for evaluation details. Device Specification [*]_

Why do we need a faster tokenizer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. rubric:: References

.. [#icu] International Components for Unicode (ICU) BreakIterator
.. [#newmm] `V. Sornlertlamvanich. Word segmentation for Thai in machine translation system. Machine Translation, NECTEC, pages 556–561, 1993. <https://www.researchgate.net/publication/243659316_Word_segmentation_for_Thai_in_machine_translation_system>`_
.. [#deepcut] `R. Kittinaradorn. DeepCut, 2017. <https://github.com/rkcosmos/deepcut>`_
.. [#multicut] `T. Lapjaturapit, K. Viriyayudhakom, and T. Theeramunkong. Multi-Candidate Word Segmentation using Bi-directional LSTM Neural Networks. pages 1–6, 2018. <https://www.researchgate.net/publication/327516094_Multi-Candidate_Word_Segmentation_using_Bi-directional_LSTM_Neural_Networks>`_
.. [#cantok] `C. Udomcharoenchaikit, P. Vateekul, and P. Boonkwan. Thai Named-Entity Recognition Using Variational Long Short-Term Memory with Conditional Random Field: Selected Revised Papers from the Joint International Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2017). pages 82–92. 2019. <https://www.researchgate.net/figure/Variational-LSTM-CRF-model-for-Thai-Named-Entity-Recognition_fig1_329766827>`_
.. [#sertis] `Sertis Corp. Thai word segmentation with bi-directional RNN <https://github.com/sertiscorp/thai-word-segmentation>`_
.. [*] For this experiment, we measured inference time on MacBook Pro (Retina, 15", Mid 2015), Intel Core i7 @ 2.2 Hz, Memory 16 GB with macOS 10.13.6.