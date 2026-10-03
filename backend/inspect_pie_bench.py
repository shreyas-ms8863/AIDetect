import csv
from pathlib import Path
from PIL import Image

manifest_path = Path("manipulation_external_test/pie_bench/pie_bench_manifest.csv")
with open(manifest_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in manifest: {len(rows)}")
for i in [0, 1, 2, 140, 220, 300, 380, 420]:
    r = rows[i]
    p = Path(r["filepath"])
    with Image.open(p) as img:
        print(f"Index {i:3d} | ID {r['sample_id']} | Cat: {r['category']} | Size: {img.size} | Mode: {img.mode}")
        print(f"  Source prompt: {r['source_prompt']}")
        print(f"  Target prompt: {r['target_prompt']}")
        print(f"  Edit action  : {r['edit_action']}")

