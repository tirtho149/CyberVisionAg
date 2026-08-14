import os
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from itertools import islice

# Get token explicitly
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("No HuggingFace token found in .env file")

print(f"Token loaded: {HF_TOKEN[:8]}...")

# ============================================================
# Load both datasets with explicit token
# ============================================================

print("\nLoading LeafNet...")
leafnet = load_dataset(
    "enalis/LeafNet",
    split="train",
    streaming=True,
    token=HF_TOKEN
)

print("Loading LeafBench...")
leafbench = load_dataset(
    "enalis/LeafBench",
    split="test",
    streaming=True,
    token=HF_TOKEN
)

# ============================================================
# Print 4 rows from LeafNet
# ============================================================

print("\n")
print("=" * 60)
print("LEAFNET — 4 SAMPLE ROWS")
print("=" * 60)

for i, sample in enumerate(islice(leafnet, 4)):
    print(f"\n--- LeafNet Row {i+1} ---")
    for key, value in sample.items():
        if key == "image":
            print(f"  {key}: <PIL Image size={value.size} mode={value.mode}>")
        elif isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")

# ============================================================
# Print 4 rows from LeafBench
# ============================================================

print("\n")
print("=" * 60)
print("LEAFBENCH — 4 SAMPLE ROWS")
print("=" * 60)

for i, sample in enumerate(islice(leafbench, 4)):
    print(f"\n--- LeafBench Row {i+1} ---")
    for key, value in sample.items():
        if key == "image":
            print(f"  {key}: <PIL Image size={value.size} mode={value.mode}>")
        elif isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from itertools import islice

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or \
           os.environ.get("HF_TOKEN")

leafnet = load_dataset(
    "enalis/LeafNet",
    split="train",
    streaming=True,
    token=HF_TOKEN
)

leafbench = load_dataset(
    "enalis/LeafBench",
    split="test",
    streaming=True,
    token=HF_TOKEN
)

# ============================================================
# LeafNet: skip healthy, find diseased examples
# Print 10 with different captions
# ============================================================

print("=" * 60)
print("LEAFNET — 10 DIVERSE DISEASED SAMPLES")
print("=" * 60)

seen_captions = set()
count = 0

for sample in islice(leafnet, 5000):
    caption = sample['caption']
    filename = sample['file_name']
    
    # Skip if we have seen this caption before
    # We want diverse examples
    caption_key = caption[:50]
    if caption_key in seen_captions:
        continue
    
    seen_captions.add(caption_key)
    count += 1
    
    print(f"\n--- LeafNet Sample {count} ---")
    print(f"  file_name : {filename}")
    print(f"  caption   : {caption}")
    print(f"  img_size  : {sample['image'].size}")
    
    if count >= 10:
        break

# ============================================================
# LeafBench: print one sample of EACH question type
# ============================================================

print("\n")
print("=" * 60)
print("LEAFBENCH — ONE SAMPLE PER QUESTION TYPE")
print("=" * 60)

seen_types = set()
count = 0

for sample in islice(leafbench, 5000):
    qtype = sample['question_type']
    
    if qtype in seen_types:
        continue
    
    seen_types.add(qtype)
    count += 1
    
    print(f"\n--- LeafBench [{qtype}] ---")
    print(f"  question_id   : {sample['question_id']}")
    print(f"  question_type : {sample['question_type']}")
    print(f"  question      : {sample['question']}")
    print(f"  A             : {sample['A']}")
    print(f"  B             : {sample['B']}")
    print(f"  C             : {sample['C']}")
    print(f"  D             : {sample['D']}")
    print(f"  answer        : {sample['answer']}")
    print(f"  img_size      : {sample['image'].size}")
    
    if len(seen_types) >= 6:
        break

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)