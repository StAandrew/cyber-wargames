import pathlib
import os
import shutil

models_dir = "models"
log_dir = "logs"
starts_with = "network"

for filename in os.listdir(models_dir):
    if filename.startswith(starts_with):
        shutil.rmtree(pathlib.Path(models_dir, filename))

for filename in os.listdir(log_dir):
    if filename.startswith(starts_with):
        shutil.rmtree(pathlib.Path(log_dir, filename))
