import json
import os
import time
from typing import Callable, Dict, NamedTuple, Union

import yaml

from attacut import logger

log = logger.get_logger(__name__)


class ModelParams(NamedTuple):
    name: str
    params: str


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        diff = self.stop - self.start
        log.info("Finished block: %s with %d seconds" % (self.name, diff))


def maybe(cond: bool, func: Callable[[], None], desc: str, verbose=0):
    if cond:
        func()
        return True
    else:
        log.info("Skipping: %s" % desc)


def maybe_create_dir(path: str):
    return maybe(
        not os.path.exists(path), lambda: os.mkdir(path), "create dir %s" % path
    )


def add_suffix_to_file_path(path: str, suffix: str) -> str:
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, ext = basename.split('.')

    return os.path.join(dirname, "%s-%s.%s" % (filename, suffix, ext))


def wc_l(path: str) -> int:
    # count total lines in a file
    s = 0
    with open(path, "r") as f:
        for _ in f:
            s += 1
    return s


def save_training_params(dir_path: str, params: ModelParams):

    dir_path = "%s/params.yml" % dir_path
    print("Saving training params to %s" % dir_path)

    params = dict(params._asdict())

    with open(dir_path, "w") as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


def load_training_params(path: str) -> ModelParams:
    with open("%s/params.yml" % path, "r") as f:
        params = yaml.load(f, Loader=yaml.BaseLoader)
        return ModelParams(**params)


def parse_model_params(ss: str) -> Dict[str, Union[int, float]]:
    params = dict()
    for pg in ss.split("|"):
        k, v = pg.split(":")

        if "." in v:
            params[k] = float(v)
        else:
            params[k] = int(v)
    return params


def load_dict(data_path: str) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        dd = json.load(f)

    log.info("loaded %d items from dict:%s" % (len(dd), data_path))

    return dd


def create_start_stop_indices(seq_lengths):
    # for old models, we might use it later???
    st_indices, sp_indices = [0], [seq_lengths[0]]

    for i, s in enumerate(seq_lengths[1:]):
        prev = sp_indices[i]
        st_indices.append(prev)
        sp_indices.append(prev + s)

    return list(zip(st_indices, sp_indices))
