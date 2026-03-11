import subprocess
import os

# GitHub repo
REPO_URL = "https://github.com/tirtho149/CyberVisionAg.git"

# directories to ignore
IGNORE_DIRS = {
    ".venv",
    ".vscode",
    "Alfalfa Diseases",
    "Corn Diseases",
    "Curated_Local_Dataset",
    "mango-leaf-disease-dataset",
    "Rye Diseases",
    "Soybean Diseases",
    "Wheat Diseases",
    "__pycache__"
}


def run(cmd):
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
    for ignore in IGNORE_DIRS:
        if ignore in path:
            return True
    return False


def collect_files():
    files = []
    for root, dirs, fs in os.walk("."):
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

        if i % 25 == 0 or i == total:
            percent = (i / total) * 100
            print(f"Added {i}/{total} files ({percent:.2f}%)")


print("\n🚀 Initializing repo\n")
run(["git", "init"])

print("\n🚀 Installing LFS\n")
run(["git", "lfs", "install"])

print("\n🚀 Tracking large files\n")
run(["git", "lfs", "track", "*.jpg"])
run(["git", "lfs", "track", "*.png"])
run(["git", "lfs", "track", "*.jpeg"])
run(["git", "lfs", "track", "*.tar.gz"])
run(["git", "lfs", "track", "*.pdf"])
run(["git", "lfs", "track", "*.xlsx"])

print("\n🚀 Adding files (datasets ignored)\n")
add_files_progress()

print("\n🚀 Committing\n")
run(["git", "commit", "-m", "Upload project (datasets ignored)"])

print("\n🚀 Setting remote\n")
run(["git", "remote", "remove", "origin"])
run(["git", "remote", "add", "origin", REPO_URL])

print("\n🚀 Setting branch\n")
run(["git", "branch", "-M", "main"])

print("\n🚀 Pushing to GitHub\n")
run(["git", "push", "-u", "origin", "main"])

print("\n✅ Upload finished!")