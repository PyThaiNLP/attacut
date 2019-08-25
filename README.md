# AttaCut
[![](https://api.travis-ci.com/heytitle/attacut.svg?token=fKbtMgf3jUWLccnyVsDw&branch=master)](https://travis-ci.com/heytitle/attacut)

**TDLR:** What is it?

## Installation

```
# only for beta version
$ pip install https://github.com/heytitle/attacut/archive/v0.5.zip
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

tokenizer = Tokenizer(model="attacut-sc")
tokenizer.tokenizer(txt)
```


## Development
We use `pipenv`.
To enable virenv
```
$ pipenv shell
```
### Data Preparation (WIP)

### Training (...)

- Local
- FloydHub

### Releasing
```
$ bumpversion [major|minor] && git push && git push --tags
```

## Related Resources
- Visualization

## Acknowledgement
- A.Te, PyThaiNLP Team, ...