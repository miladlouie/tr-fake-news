import csv
from pathlib import Path

BASE = Path("data/raw/Clean")

fake_dir = BASE / "Zaytung"
real_dir = BASE / "Hurriyet"

output = Path("data/turkish_fake_news.csv")
output.parent.mkdir(exist_ok=True)

rows = []

# Fake news (Zaytung)
for file in fake_dir.glob("*.txt"):
    text = file.read_text(encoding="utf-8").strip()
    if text:
        rows.append([text, 0])  # FAKE = 0

# Real news (Hurriyet)
for file in real_dir.glob("*.txt"):
    text = file.read_text(encoding="utf-8").strip()
    if text:
        rows.append([text, 1])  # REAL = 1

print(f"Loaded: {len(rows)} samples")

# Save CSV
with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Saved → {output}")
