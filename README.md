# AttaCut
[![Build Status](https://travis-ci.org/PyThaiNLP/attacut.svg?branch=master)](https://travis-ci.org/PyThaiNLP/attacut)
[![](https://img.shields.io/badge/-presentation-informational)](https://drive.google.com/file/d/16AUNZv1HXVmERgryfBf4JpCo1QrQyHHE/view?usp=sharing)
![](https://img.shields.io/badge/doi-WIP-informational)

<div align="center">
    <img src="https://i.imgur.com/8yMq7IB.png" width="700px"/>
    <br/>
    <b>TLDR:</b> 
3-Layer dilated CNN on character and syllable features
</div>

## Installation

```
# only for beta version
$ pip install attacut==0.0.3.dev0
```

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

### Higher-Level Inferface
aka. module importing
```
from attacut import Tokenizer

atta = Tokenizer(model="attacut-sc")
atta.tokenizer(txt)
```

## Development
Please refer to [DEVELOPMENT.md](./docs/DEVELOPMENT.md)

## Related Resources
- [Tokenization Visualization][tovis]
- [Thai Tokenizer Dockers][docker]

## Acknowledgements
- This repository was initially done by [Pattarawat Chormai][pat], while interning at [Dr. Attapol Thamrongrattanarit's NLP Lab][ate], Chulalongkorn University, Bangkok, Thailand.
- Many thanks to my collegeus at Dr. Attapol's lab, PyThaiNLP team, [Ekapol Chuangsuwanich ][ake], [Noom][noom], and [Can][can] for comments and feedback.


[pat]: http://pat.chormai.org
[ate]: https://attapol.github.io/lab.html
[noom]: https://github.com/Ekkalak-T
[can]: https://github.com/c4n
[ake]: https://github.com/ekapolc
[tovis]: https://pythainlp.github.io/tokenization-benchmark-visualization/
[docker]: https://github.com/PyThaiNLP/docker-thai-tokenizers