import kagglehub
import os
import shutil

# Download latest version to KaggleHub's cache
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Downloaded dataset path:", path)

# Define your project directory (current directory)
project_dir = os.path.dirname(os.path.abspath(__file__))   # <-- script's directory
# If running in Jupyter/Notebook, use:
# project_dir = os.getcwd()

# Move/copy dataset files to project directory
for file_name in os.listdir(path):
    src = os.path.join(path, file_name)
    dst = os.path.join(project_dir, file_name)

    if not os.path.exists(dst):  # avoid overwriting if already exists
        shutil.copy2(src, dst)

print("Files copied to project directory:", project_dir)