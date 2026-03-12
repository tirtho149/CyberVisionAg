import subprocess
import os

REPO_URL = "https://github.com/tirtho149/CyberVisionAg.git"
BRANCH = "updated"

IGNORE_DIRS = {
    ".venv",
    ".vscode",
    "__pycache__",
    "Alfalfa Diseases",
    "Corn Diseases",
    "Curated_Dataset",
    "mango-leaf-disease-dataset",
    "Rye Diseases",
    "Soybean Diseases",
    "Wheat Diseases",
    "/home/user/Desktop/CyberAG/CDDM-images",
    "/home/user/Desktop/CyberAG/InternalData"
    "data"
    
}

IGNORE_FILES = {
    ".env"
}


def run(cmd):
    """Run command and stream output"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

    process.wait()


def should_ignore(path):
    parts = set(path.split(os.sep))

    if parts & IGNORE_DIRS:
        return True

    if os.path.basename(path) in IGNORE_FILES:
        return True

    return False


def collect_files():
    files = []

    for root, dirs, fs in os.walk("."):

        # remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for f in fs:
            path = os.path.join(root, f)

            if not should_ignore(path):
                files.append(path)

    return files


def add_files_progress():
    files = collect_files()
    total = len(files)

    print(f"\n📦 Files to upload: {total}\n")

    for i, file in enumerate(files, 1):

        subprocess.run(
            ["git", "add", file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if i % 50 == 0 or i == total:
            percent = (i / total) * 100
            print(f"Added {i}/{total} files ({percent:.2f}%)")


print("\n🚀 Checking repo\n")

if not os.path.exists(".git"):
    run(["git", "init"])

print("\n🚀 Installing Git LFS\n")
run(["git", "lfs", "install"])

print("\n🚀 Tracking large files\n")
run(["git", "lfs", "track", "*.jpg"])
run(["git", "lfs", "track", "*.jpeg"])
run(["git", "lfs", "track", "*.png"])
run(["git", "lfs", "track", "*.tar.gz"])
run(["git", "lfs", "track", "*.pdf"])
run(["git", "lfs", "track", "*.xlsx"])

print("\n🚀 Adding LFS attributes\n")
run(["git", "add", ".gitattributes"])

print("\n🚀 Adding project files\n")
add_files_progress()

print("\n🚀 Committing\n")
run(["git", "commit", "-m", "Upload project with Git LFS (datasets ignored)"])

print("\n🚀 Configuring remote\n")

result = subprocess.run(
    ["git", "remote"],
    capture_output=True,
    text=True
)

if "origin" in result.stdout:
    run(["git", "remote", "set-url", "origin", REPO_URL])
else:
    run(["git", "remote", "add", "origin", REPO_URL])

print("\n🚀 Switching branch\n")
run(["git", "checkout", "-B", BRANCH])

print("\n🚀 Pushing to GitHub\n")
run(["git", "push", "-u", "origin", BRANCH])

print("\n✅ Upload finished!")