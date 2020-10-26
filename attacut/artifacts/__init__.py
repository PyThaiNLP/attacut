import os

from attacut import logger

log = logger.get_logger(__name__)

artifact_dir = os.path.dirname(__file__)

model_list = [name for name in os.listdir(artifact_dir) if os.path.isdir(os.path.join(artifact_dir,name))]


def get_path(name: str) -> str:
    if name in model_list:
        return f"{artifact_dir}/{name}"
    else:
        # if name isn't in the list, then it's a custom model
        log.info("model_path: %s" % name)
        return name
