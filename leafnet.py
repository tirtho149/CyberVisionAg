# ============================================================
# Load Hugging Face token via dotenv + stream 100 samples only
# ============================================================

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Make token visible to HF libraries
if "HUGGINGFACE_HUB_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]

from datasets import load_dataset
from PIL import Image
import numpy as np
import hashlib
import random
from itertools import islice

# --------------------
# Config
# --------------------
SAMPLE_SIZE = 10000
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --------------------
# Load datasets (STREAMING)
# --------------------
leafnet = load_dataset(
    "enalis/LeafNet",
    split="train",
    streaming=True
)

leafbench = load_dataset(
    "enalis/LeafBench",
    split="test",
    streaming=True
)

# --------------------
# Image utilities
# --------------------
def normalize_image(img, size=IMAGE_SIZE):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img

def hash_image(img):
    return hashlib.sha256(normalize_image(img).tobytes()).hexdigest()

# --------------------
# Sample ONLY 100 (no full download)
# --------------------
leafnet_samples = list(islice(leafnet, SAMPLE_SIZE))
leafbench_samples = list(islice(leafbench, SAMPLE_SIZE))

# --------------------
# Hash & overlap
# --------------------
leafnet_hashes = {hash_image(s["image"]) for s in leafnet_samples}

overlap = [
    s for s in leafbench_samples
    if hash_image(s["image"]) in leafnet_hashes
]

# --------------------
# Results
# --------------------
print(f"LeafNet samples      : {len(leafnet_samples)}")
print(f"LeafBench samples    : {len(leafbench_samples)}")
print(f"Overlapping images   : {len(overlap)}")
print(f"Overlap rate         : {len(overlap)/len(leafbench_samples):.2%}")