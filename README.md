# AttaCut: Fast and Reasonably Accurate Word Tokenizer for Thai

[![Build Status](https://travis-ci.org/PyThaiNLP/attacut.svg?branch=master)](https://travis-ci.org/PyThaiNLP/attacut)
[![Build status](https://ci.appveyor.com/api/projects/status/ittfnb2pyg95kpxk/branch/master?svg=true)](https://ci.appveyor.com/project/wannaphongcom/attacut/branch/master)
[![](https://img.shields.io/badge/-presentation-informational)](https://drive.google.com/file/d/16AUNZv1HXVmERgryfBf4JpCo1QrQyHHE/view?usp=sharing)
![](https://img.shields.io/badge/doi-WIP-informational)

## How does AttaCut look like?

<div align="center">
    <img src="https://i.imgur.com/8yMq7IB.png" width="700px"/>
    <br/>
    <b>TL;DR:</b> 
3-Layer Dilated CNN on syllable and character features. It’s <b>6x faster</b> than DeepCut (SOTA) while its WL-f1 on BEST is <b>91%</b>, only 2% lower.
</div>

## Installation

```
$ pip install attacut
```

**Remarks:** Windows users need to install **PyTorch** before the command above.
Please consult [PyTorch.org](https://pytorch.org) for more details.

## Usage

### Command-Line Interface
```
$ attacut-cli -h
AttaCut: Fast and Reasonably Accurate Tokenizer for Thai

Usage:
  attacut-cli <src> [--dest=<dest>] [--model=<model>]
  attacut-cli (-h | --help)

Options:
  -h --help         Show this screen.
  --model=<model>   Model to be used [default: attacut-sc].
  --dest=<dest>     If not specified, it'll be <src>-tokenized-by-<model>.txt
```

### High-Level API 
```
from attacut import tokenize, Tokenizer

# tokenize `txt` using our best model `attacut-sc`
words = tokenize(txt)

# alternatively, an AttaCut tokenizer might be instantiated directly, allowing
# one to specify whether to use `attacut-sc` or `attacut-c`.
atta = Tokenizer(model="attacut-sc")
words = atta.tokenize(txt)
```

## Benchmark Results

Belows are brief summaries. More details can be found on [our benchmarking page](https://pythainlp.github.io/attacut/benchmark.html).


### Tokenization Quality
![](https://pythainlp.github.io/attacut/_images/quality-benchmark-in-of-domain.png)

### Speed
![](https://pythainlp.github.io/attacut/_images/speed-benchmark-ec2.png)


## Retraining on Custom Dataset
Please refer to [our retraining page](https://pythainlp.github.io/attacut/)

## Related Resources
- [Tokenization Visualization][tovis]
- [Thai Tokenizer Dockers][docker]

## Acknowledgements
This repository was initially done by [Pattarawat Chormai][pat], while interning at [Dr. Attapol Thamrongrattanarit's NLP Lab][ate], Chulalongkorn University, Bangkok, Thailand.
Many people have involed in this project. Complete list of names can be found on [Acknowledgement](https://pythainlp.github.io/attacut/acknowledgement.html).


[pat]: http://pat.chormai.org
[ate]: https://attapol.github.io/lab.html
[noom]: https://github.com/Ekkalak-T
[can]: https://github.com/c4n
[ake]: https://github.com/ekapolc
[tovis]: https://pythainlp.github.io/tokenization-benchmark-visualization/
[docker]: https://github.com/PyThaiNLP/docker-thai-tokenizers
