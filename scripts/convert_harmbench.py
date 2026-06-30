import csv
import json
from pathlib import Path

def convert_csv_to_json(csv_path: Path, json_path: Path):
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                "id": row["BehaviorID"],
                "goal": row["Behavior"],
                "category": row["SemanticCategory"],
                "tags": row["Tags"]
            })

    with open(json_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    print(f"Converted {csv_path} to {json_path}")

base_dir = Path("data/harmbench")
csv_dir = base_dir / "behavior_datasets"
json_dir = base_dir / "json_datasets"

# Convert relevant files
csv_files = ["harmbench_behaviors_text_test.csv", "harmbench_behaviors_text_val.csv", "harmbench_behaviors_text_all.csv"]

for csv_file in csv_files:
    convert_csv_to_json(csv_dir / csv_file, json_dir / csv_file.replace(".csv", ".json"))
