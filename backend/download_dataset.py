"""Download Flickr8k dataset from Kaggle."""
import os
import shutil
import kagglehub

def download():
    print("Downloading Flickr8k...")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print(f"Path: {path}")
    if os.path.exists(os.path.join(path, "Images")):
        src = path
    else:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        src = os.path.join(path, subdirs[0]) if subdirs and os.path.exists(os.path.join(path, subdirs[0], "Images")) else path
    target = "./flickr8k"
    if not os.path.exists(target):
        shutil.copytree(src, target)
    print(f"Dataset ready at {target}")

if __name__ == "__main__":
    download()
