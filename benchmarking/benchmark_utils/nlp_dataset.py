import json
from pathlib import Path
from torch.utils.data import Dataset

class NLPGoalDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.goals = json.load(f)

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, idx):
        # Return goal, empty label (or as needed for NLP), and metadata
        item = self.goals[idx]
        return item["goal"], None, {"id": item["id"], "category": item.get("category")}
