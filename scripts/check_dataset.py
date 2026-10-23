
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("Dataset folder:", DATA_DIR.resolve())
print("Total images:", len(images))
print("\nTop-level folders:")

for item in sorted(DATA_DIR.iterdir()):
    if item.is_dir():
        print(f"  Folder: {item.name}")
    elif item.is_file():
        print(f"  File: {item.name}")

print("\nSample image filenames:")
for p in images[:20]:
    print(" ", p.relative_to(DATA_DIR))

print("\nFilename keyword counts:")
counts = Counter()
for p in images:
    name = p.name.lower()
    if "spoof" in name or "fake" in name:
        counts["possible spoof"] += 1
    elif "real" in name or "live" in name:
        counts["possible real"] += 1
    else:
        counts["unknown"] += 1

for label, count in counts.items():
    print(f"  {label}: {count}")

