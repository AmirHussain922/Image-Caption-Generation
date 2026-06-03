
import os
import pickle
import requests

WORKING_DIR = os.environ.get("WORKING_DIR", "./working")
BASE_DIR = os.environ.get("BASE_DIR", "./flickr8k")

# Load actual captions
captions_file = os.path.join(BASE_DIR, "captions.txt")
actual_captions = {}

with open(captions_file, "r") as f:
    next(f)
    for line in f:
        tokens = line.strip().split(",")
        if len(tokens) < 2:
            continue
        image_id = tokens[0].split(".")[0]
        caption = " ".join(tokens[1:])
        if image_id not in actual_captions:
            actual_captions[image_id] = []
        actual_captions[image_id].append(caption.lower())

# Test image IDs
test_image_ids = [
    "1000268201_693b08cb0e",
    "1001773457_577c3a7d70",
    "1002674143_1b742ab4b8",
    "1003163366_44323f5815"
]

print("Testing caption accuracy...")
print("=" * 80)

for img_id in test_image_ids:
    print(f"\nImage ID: {img_id}")
    
    # Get actual captions
    actual = actual_captions.get(img_id, [])
    print("Actual captions:")
    for i, cap in enumerate(actual):
        print(f"  {i+1}. {cap}")
    
    # Get predicted caption
    try:
        response = requests.get(f"http://localhost:8000/sample/{img_id}")
        response.raise_for_status()
        data = response.json()
        predicted = data["caption"].lower()
        print(f"Predicted caption: {predicted}")
    except Exception as e:
        print(f"Error getting prediction: {e}")

print("\n" + "=" * 80)
print("Test complete!")
