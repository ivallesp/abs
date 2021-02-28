import os


def get_model_folder(alias):
    path = os.path.join("models", alias)
    if not os.path.exists(path):
        os.makedirs(path)
    return path