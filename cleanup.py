import os
import pathlib
import shutil

from config import models_dir, log_dir, starts_with


for filename in os.listdir(models_dir):
    if filename.startswith(starts_with):
        shutil.rmtree(pathlib.Path(models_dir, filename))

for filename in os.listdir(log_dir):
    if filename.startswith(starts_with):
        shutil.rmtree(pathlib.Path(log_dir, filename))
