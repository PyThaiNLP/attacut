import os

artifact_dir = os.path.dirname(__file__)

def get_path(name):
    # todo: check if the artifact exits
    return f"{artifact_dir}/{name}"
