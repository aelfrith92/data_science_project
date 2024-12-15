import os

ASSETS_DIR = "./assets/plots"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)