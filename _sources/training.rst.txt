Retraining
----------

Project Overview
^^^^^^^^^^^^^^^^

AttaCut project is structured into several submodules. Four of them are
important and worth knowing for further customization beyond what
we have provided.

1. **Models** contains definitions of models.
2. **Dataloaders** contains functionalities to process data for a particular model.
3. **Preprocessing** includes methods for cleaning data.
4. **Utils** contains other helper functions.

Training Data Format
^^^^^^^^^^^^^^^^^^^^

To train an AttaCut model, one needs to prepare data as follow:

For AttaCut-SC
""""""""""""""
1. Character Dictionary: mapping from a character to an index
2. Syllable Dicitonary:  mapping from a syllable to an index
3. | Training and Validation sets. These sets have to be in the format below:

    .. code-block:: bash

        1000100101::CH_IX CH_IX CH_IX ...::SY_IX SY_IX ...

+-----------------+---------------------------------------------------------------------------------------+
| **Explanation** |                                    **Description**                                    |
+-----------------+---------------------------------------------------------------------------------------+
|    100010010    | sequence of labels. `1` indicates a starting-word character.                          |
+-----------------+---------------------------------------------------------------------------------------+
|    CH_IX ...    | sequence of character indices                                                         |
+-----------------+---------------------------------------------------------------------------------------+
|    SY_IX ...    | sequence of syllable indices. Characters in the same syllable have the same **SY_IX** |
+-----------------+---------------------------------------------------------------------------------------+

   | Each line could be a line in your original text.

With these ingredients, one has to create a directory:

.. code-block:: bash

    # ls -la ./some-dataset
    characters.json
    syllables.json
    training.txt
    val.txt

Our AttaCut-SC training data can be found here:

- https://www.floydhub.com/pattt/datasets/best-syllable-crf-and-character-seq-feature-sampling-0/3
- https://www.floydhub.com/pattt/datasets/character-dict-min-freq-10/2
- https://www.floydhub.com/pattt/datasets/syllable-crf-dict-min-freq-10/2


For AttaCut-C
"""""""""""""
Every detail is similar to the preparation of AttaCut-SC, except that we do need
the syllable dictionary and syllable indices **(SY_IX SY_IX ...)**.

Our AttaCut-C training data can be found here:

- https://www.floydhub.com/pattt/datasets/best-character-seq-feature-sampling-0/1
- https://www.floydhub.com/pattt/datasets/character-dict-min-freq-10/2


How to Retrain on Custom Dataset?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Our training script is provided in `./scripts/train.py`. Several options can be specified when calling the script.

This is an example of how we use it:

.. code-block:: bash

    $ python ./scripts/train.py --model-name seq_sy_ch_conv_concat \ # seq_sy_ch_conv_concat = attacut-sc
        --model-params "embc:8|embs:8|conv:8|l1:6|do:0.1"' \ # emb{c,s} are embedding dimensions
        --data-dir ./some-data  \
        --output-dir ./sink/model-xx  \
        --epoch 10 \
        --batch-size 1024 \
        --lr 0.001 \
        --lr-schedule "step:5|gamma:0.5"


AttaCut's training code is primarily built to be used on
`FloydHub <https://www.floydhub.com/pattt/projects/attacut>`_. Our training jobs
for the released models are:

- AttaCut-SC: https://www.floydhub.com/pattt/projects/attacut/50
- AttaCut-C: https://www.floydhub.com/pattt/projects/attacut/42


Please let us know if you have any further questions.

Happy coding and less overfitting! 🤪
