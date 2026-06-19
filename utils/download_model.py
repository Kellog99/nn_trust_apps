import datetime
import json
from pathlib import Path

import timm
from timm.models import PretrainedCfg
from tqdm import tqdm

if __name__ == "__main__":
    path: Path = Path("~/Desktop/StableAI/model_repository").expanduser()
    for id in tqdm(timm.list_models()[:100]):
        config: PretrainedCfg = timm.get_pretrained_cfg(id)
        if config is None:
            continue
        out = {
            "id": config.architecture,
            "name": config.architecture,
            "date": datetime.date.today().isoformat(),
            "task": "classification",
            "domain": "cv",
            'std': config.std,
            "mean": config.mean,
            "input_dimensionality": config.input_size,
            "model_type": "timm",
            "description": config.description,
            "num_classes": config.num_classes
        }
        model_repo: Path = path / config.architecture
        model_repo.mkdir(parents=True, exist_ok=True)
        with open(model_repo / "info.json", "w") as f:
            json.dump(out, f)
