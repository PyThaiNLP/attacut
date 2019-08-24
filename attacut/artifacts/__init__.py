import os

artifact_dir = os.path.dirname(__file__)

def get_path(name:str) -> str:
    if name in ['attacut-c', 'attacut-sc']:
        return f"{artifact_dir}/{name}"
    else:
        # if name isn't in the list, then it's a custom model
        return name
