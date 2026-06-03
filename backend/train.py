"""
Image Caption Generator - Training Script
Auto-downloads Flickr8k dataset via kagglehub if missing.
"""

import os
import re
import pickle
import shutil
import numpy as np
from tqdm.std import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# ============================
# CHECK GPU AVAILABILITY
# ============================
print("Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

# ============================
# CONFIG
# ============================
BASE_DIR = os.environ.get("BASE_DIR", "./flickr8k")
WORKING_DIR = os.environ.get("WORKING_DIR", "./working")
TRAINED_PICS_DIR = os.path.join(os.path.dirname(os.path.dirname(WORKING_DIR)), "trained pictures")
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(TRAINED_PICS_DIR, exist_ok=True)

def ensure_dataset():
    images_dir = os.path.join(BASE_DIR, "Images")
    captions_file = os.path.join(BASE_DIR, "captions.txt")
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("Dataset found locally.")
        return
    if not KAGGLEHUB_AVAILABLE:
        raise FileNotFoundError("Dataset missing and kagglehub not installed. pip install kagglehub")
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print(f"Downloaded to: {path}")
    if os.path.exists(os.path.join(path, "Images")):
        src = path
    else:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        src = os.path.join(path, subdirs[0]) if subdirs and os.path.exists(os.path.join(path, subdirs[0], "Images")) else path
    if not os.path.exists(BASE_DIR):
        shutil.copytree(src, BASE_DIR)
    print(f"Dataset ready at {BASE_DIR}")

ensure_dataset()

# ============================
# 1. LOAD EXISTING FEATURES & TOKENIZER IF AVAILABLE
# ============================
features_path = os.path.join(WORKING_DIR, "features.pkl")
tokenizer_path = os.path.join(WORKING_DIR, "tokenizer.pkl")

if os.path.exists(features_path):
    print("Loading existing features...")
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    print(f"Features loaded: {len(features)} images")
else:
    print("Loading VGG16...")
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

    features = {}
    directory = os.path.join(BASE_DIR, "Images")
    print("Extracting features...")
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        image_id = img_name.split(".")[0]
        features[image_id] = feature

    pickle.dump(features, open(features_path, "wb"))
    print(f"Features saved: {len(features)} images")

if os.path.exists(tokenizer_path):
    print("Loading existing tokenizer...")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    # Also need mapping and max_length, let's load those from scratch quickly
    with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
        next(f)
        captions_doc = f.read()

    mapping = {}
    for line in captions_doc.split("\n"):
        tokens = line.split(",")
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                caption = re.sub(r"[^A-Za-z\s]", "", caption)
                caption = re.sub(r"\s+", " ", caption).strip()
                caption = "startseq " + " ".join([w for w in caption.split() if len(w) > 1]) + " endseq"
                captions[i] = caption

    clean(mapping)
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"Tokenizer loaded: vocab size {vocab_size}, max length {max_length}")
else:
    # ============================
    # 2. LOAD CAPTIONS
    # ============================
    with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
        next(f)
        captions_doc = f.read()

    mapping = {}
    for line in tqdm(captions_doc.split("\n")):
        tokens = line.split(",")
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    print(f"Loaded {len(mapping)} images")

    # ============================
    # 3. PREPROCESS TEXT
    # ============================
    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                caption = re.sub(r"[^A-Za-z\s]", "", caption)
                caption = re.sub(r"\s+", " ", caption).strip()
                caption = "startseq " + " ".join([w for w in caption.split() if len(w) > 1]) + " endseq"
                captions[i] = caption

    clean(mapping)

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    print(f"Total captions: {len(all_captions)}")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocab size: {vocab_size}")

    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"Max length: {max_length}")

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

# ============================
# 4. SPLIT
# ============================
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]
print(f"Train: {len(train)}, Test: {len(test)}")

# ============================
# 5. GENERATOR
# ============================
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# ============================
# 6. BUILD MODEL
# ============================
inputs1 = Input(shape=(4096,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam")
print(model.summary())
# plot_model(model, to_file=os.path.join(WORKING_DIR, "model_architecture.png"), show_shapes=True)  # Commented out - requires pydot

# ============================
# 7. TRAIN
# ============================
epochs = 10
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    print(f"\nEpoch {i+1}/{epochs}")
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save(os.path.join(WORKING_DIR, "best_model.keras"))
print("Model saved!")

# ============================
# COPY SAMPLE IMAGES TO TRAINED PICTURES FOLDER
# ============================
print(f"\nCopying sample images to {TRAINED_PICS_DIR}")
sample_image_names = [
    "1000268201_693b08cb0e.jpg",
    "1001773457_577c3a7d70.jpg",
    "1002674143_1b742ab4b8.jpg",
    "1003163366_44323f5815.jpg"
]

for img_name in sample_image_names:
    src = os.path.join(BASE_DIR, "Images", img_name)
    dst = os.path.join(TRAINED_PICS_DIR, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {img_name}")

# ============================
# 8. EVALUATE
# ============================
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text

actual, predicted = list(), list()
for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual.append([c.split() for c in captions])
    predicted.append(y_pred.split())

print("\nBLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# ============================
# 9. VISUALIZE
# ============================
def generate_caption(image_name):
    image_id = image_name.split(".")[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print("\n----- Actual -----")
    for c in captions:
        print(c)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print("----- Predicted -----")
    print(y_pred)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# generate_caption("1001773457_577c3a7d70.jpg")
