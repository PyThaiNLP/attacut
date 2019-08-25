import os
import numpy as np
import yaml
import time
import json

from attacut import logger

log = logger.get_logger(__name__)

def maybe(cond, func, desc, verbose=0):
    if cond:
        func()
        return True
    else:
        print("Skipping: %s" % desc)

def maybe_create_dir(path):
    return maybe(
        not os.path.exists(path),
        lambda : os.mkdir(path),
        "create dir %s" % path
    )

def add_suffix_to_file_path(path, suffix):
    components = path.split("/")
    filename, ext = components[-1].split(".")

    components[-1] = "%s-%s.%s" % (filename, suffix, ext)

    return "/".join(components)

def build_window_indices(window_size=1):
    return np.arange(2*window_size + 1 ) - window_size

def padding_array(arr, padding_value=-1, padding_size=1):
    return np.array(
        [padding_value]*padding_size
        + arr 
        + [padding_value]*padding_size
    )

def wc_l(path):
    s = 0
    with open(path, "r") as f:
        for l in f:
            s += 1
    return s

def save_training_params(dir_path, params):

    dir_path = "%s/params.yml" % dir_path
    print("Saving training params to %s" % dir_path)

    with open(dir_path, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)

def load_training_params(path):
    with open("%s/params.yml" % path, "r") as f:
        return yaml.load(f, Loader=yaml.BaseLoader)

def parse_model_params(ss):
    params = dict()
    for pg in ss.split("|"):
        k, v = pg.split(":")

        if "." in v:
            params[k] = float(v)
        else:
            params[k] = int(v)
    return params

def create_start_stop_indices(seq_lengths):
    # todo: remove it?
    st_indices, sp_indices = [0], [seq_lengths[0]]

    for i, s in enumerate(seq_lengths[1:]):
        prev = sp_indices[i]
        st_indices.append(prev)
        sp_indices.append(prev + s)

    return list(zip(st_indices, sp_indices))

def load_dict(data_path):
    with open(data_path, "r") as f:
        dd = json.load(f)

    log.info("loaded %d items from dict:%s" % (len(dd), data_path))

    return dd

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        diff = self.stop - self.start
        print("Finished block: %s with %d seconds" %  (self.name, diff))