
import pickle
import os

WORKING_DIR = os.environ.get("WORKING_DIR", "./working")
features_path = os.path.join(WORKING_DIR, "features.pkl")

with open(features_path, "rb") as f:
    features = pickle.load(f)

print("First 10 image IDs:")
for i, img_id in enumerate(list(features.keys())[:10]):
    print(f"{i+1}. {img_id}")
