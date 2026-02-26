"""
Print Directory Structure
=========================
Prints the full folder/file layout of any directory.

USAGE:
    python print_directory_structure.py

Just set YOUR_DIRECTORY below to your local data path.
"""

import os

# ============================================================================
# ★ SET YOUR DIRECTORY PATH HERE ★
# ============================================================================

YOUR_DIRECTORY = "/home/user/Desktop/Disease Images for Arti"
                       # Examples:
                       #   Windows : "C:/Users/yourname/mydata"
                       #   Linux   : "/home/yourname/mydata"
                       #   Mac     : "/Users/yourname/mydata"

# Show image files too? (True = show all files, False = folders only)
SHOW_FILES = True

# Max depth to scan (None = unlimited)
MAX_DEPTH = 5

# ============================================================================
# PRINT FUNCTION
# ============================================================================

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def print_structure(directory, prefix="", depth=0):
    """Recursively print the directory tree."""

    if MAX_DEPTH is not None and depth > MAX_DEPTH:
        print(f"{prefix}... (max depth reached)")
        return

    if not os.path.isdir(directory):
        print(f"\n✗ ERROR: Directory not found → {os.path.abspath(directory)}")
        return

    items = sorted(os.listdir(directory))
    folders = [i for i in items if os.path.isdir(os.path.join(directory, i))]
    files   = [i for i in items if os.path.isfile(os.path.join(directory, i))]
    images  = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
    others  = [f for f in files if not f.lower().endswith(IMAGE_EXTENSIONS)]

    # Print folders recursively
    for idx, folder in enumerate(folders):
        is_last_folder = (idx == len(folders) - 1) and (not SHOW_FILES or not files)
        connector = "└── " if is_last_folder else "├── "
        child_prefix = prefix + ("    " if is_last_folder else "│   ")

        folder_path = os.path.join(directory, folder)
        sub_items   = os.listdir(folder_path)
        sub_folders = [s for s in sub_items if os.path.isdir(os.path.join(folder_path, s))]
        sub_images  = [s for s in sub_items if s.lower().endswith(IMAGE_EXTENSIONS)]

        # Annotate with counts
        annotation = ""
        if sub_images and not sub_folders:
            annotation = f"  [{len(sub_images)} images]"
        elif sub_folders:
            annotation = f"  [{len(sub_folders)} subfolders]"

        print(f"{prefix}{connector}{folder}/{annotation}")
        print_structure(folder_path, child_prefix, depth + 1)

    # Print files
    if SHOW_FILES and files:
        if images:
            is_last = not others
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}... {len(images)} image files "
                  f"({', '.join(set(os.path.splitext(f)[1].lower() for f in images))})")
        for idx, f in enumerate(others):
            connector = "└── " if idx == len(others) - 1 else "├── "
            print(f"{prefix}{connector}{f}")


def count_summary(directory):
    """Print a summary: total folders, total images, per-class counts."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_folders = 0
    total_images  = 0
    class_counts  = {}

    for root, dirs, files in os.walk(directory):
        dirs.sort()
        rel = os.path.relpath(root, directory)
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
        if images:
            total_images  += len(images)
            total_folders += 1
            class_counts[rel] = len(images)

    print(f"  Total folders with images : {total_folders}")
    print(f"  Total image files         : {total_images}")
    print(f"\n  {'Folder':<50} {'Images':>6}")
    print(f"  {'-'*57}")
    for folder, count in sorted(class_counts.items()):
        status = "✓ OK" if count >= 15 else f"⚠ only {count} (need 15 for 5 train + 10 test)"
        print(f"  {folder:<50} {count:>6}   {status}")
    print(f"  {'-'*57}")
    print(f"  {'TOTAL':<50} {total_images:>6}")


# ============================================================================
# RUN
# ============================================================================

abs_path = os.path.abspath(YOUR_DIRECTORY)
print(f"\n{'='*60}")
print(f"DIRECTORY STRUCTURE")
print(f"{'='*60}")
print(f"Root: {abs_path}\n")

print_structure(YOUR_DIRECTORY)
count_summary(YOUR_DIRECTORY)