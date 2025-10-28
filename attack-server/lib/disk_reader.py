from pathlib import Path
from typing import Optional, Tuple, Union
import logging

def find_model_and_task_dir(
    base_dir: Union[str, Path],
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Find the model directory and the task-id directory.

    Args:
        base_dir: root folder that contains 'benchmark_out' (e.g. '/path/to/NN_TRUST_APPS/attack-server')
        dataset: dataset name (e.g. 'animals')
        model: model name (e.g. 'resnet50')
        task_id: exact task folder name (timestamp folder, e.g. '20251028T113502')

    Returns:
        (model_dir_path, task_dir_path) as strings

    Behavior:
        - If task_id is provided:
            - If dataset+model provided: verifies and returns benchmark_out/task_id/dataset/model
            - If dataset/model not provided: returns the first dataset/model found under the task folder
        - If task_id is NOT provided:
            - dataset+model must be provided; searches across all task folders and returns the latest task
              (by modification time) that contains dataset/model.

    Raises:
        FileNotFoundError when required folders cannot be found or inputs are insufficient.
        ValueError for invalid inputs.
    """
    try:
        base = Path(base_dir).expanduser().resolve()
        bench = base 
        if not bench.exists() or not bench.is_dir():
            raise FileNotFoundError(f"'benchmark_out' not found under base_dir: {base}")

        # Helper to verify model inside a specific task directory
        def _find_in_task(task_dir: Path) -> Optional[Path]:
            """Return model Path if found inside task_dir, else None."""
            if dataset and model:
                candidate = task_dir / dataset / model
                if candidate.is_dir():
                    return candidate
                # try recursive search for model where parent is dataset
                for p in task_dir.rglob(model):
                    if p.is_dir() and any(parent.name == dataset for parent in p.parents):
                        return p
                return None
            else:
                # dataset/model not provided -> return first dataset/model found under task_dir
                for ds in sorted(task_dir.iterdir()):
                    if ds.is_dir():
                        for m in sorted(ds.iterdir()):
                            if m.is_dir():
                                return m
                return None

        # If task_id provided: operate inside that folder
        if task_id:
            task_dir = bench / task_id
            if not task_dir.exists() or not task_dir.is_dir():
                raise FileNotFoundError(f"Task id folder not found: {task_dir}")
            model_dir = _find_in_task(task_dir)
            if model_dir is None:
                if dataset and model:
                    raise FileNotFoundError(f"Dataset/model '{dataset}/{model}' not found under task {task_id}")
                else:
                    raise FileNotFoundError(f"No dataset/model found under task {task_id}")
            return str(model_dir), str(task_dir)

        # No task_id provided -> dataset+model must be provided
        if not (dataset and model):
            raise ValueError("Either task_id must be provided, or both dataset and model must be provided.")

        # Iterate over all task directories and collect candidates
        candidates = []
        for td in (d for d in bench.iterdir() if d.is_dir()):
            found = _find_in_task(td)
            if found:
                candidates.append((td.stat().st_mtime, td, found))

        if not candidates:
            raise FileNotFoundError(f"No task folder in '{bench}' contains dataset/model '{dataset}/{model}'")

        # choose latest task by modification time
        latest = max(candidates, key=lambda t: t[0])
        _, task_dir, model_dir = latest
        return str(model_dir), str(task_dir)
    except Exception as e:
        logging.error(f"Error finding model/task directories: {e}")
        raise


from pathlib import Path
import json
from typing import List, Dict, Any, Optional

from pathlib import Path
import json
from typing import List, Dict, Any, Optional

def collect_dataset_aggregates_with_info(
    base_dir: str,
    dataset: str,
    *,
    filenames_pattern: str = "aggregate*.json",
) -> List[Dict[str, Any]]:
    """
    Search each timestamp (task) folder under base_dir for files matching
    filenames_pattern inside the given dataset folder. For each unique
    relative dataset-path (e.g. model folder under dataset), keep only the
    aggregate from the *latest* task folder (by modification time).

    Returns a list of aggregate dicts, augmented with:
      - "__task_dir": task folder name (string)
      - "__task_mtime": task folder mtime (float)
      - "__agg_path": full path to the aggregate file (string)
    """
    base = Path(base_dir).expanduser().resolve()
    bench = base
    if not bench.exists() or not bench.is_dir():
        raise FileNotFoundError(f"'benchmark_out' not found under base_dir: {base}")

    # key -> tuple(task_mtime, result_dict)
    latest_per_relpath: Dict[str, tuple[float, Dict[str, Any]]] = {}

    # iterate over task folders
    # we don't rely on iteration order; we'll compare mtimes explicitly
    for task_dir in (p for p in bench.iterdir() if p.is_dir()):
        dataset_dir = task_dir / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            # dataset not present in this task folder
            continue

        # mtime of task folder (used to choose latest)
        try:
            task_mtime = float(task_dir.stat().st_mtime)
        except Exception:
            # fallback: zero if we cannot stat
            task_mtime = 0.0

        # search for aggregate files under the dataset directory (including model subdirs)
        for agg_file in dataset_dir.rglob(filenames_pattern):
            if not agg_file.is_file():
                continue
            try:
                with agg_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                # skip invalid json aggregate file
                continue

            # try to read info.json in the same directory as the aggregate file
            info_path = agg_file.parent / "info.json"
            if info_path.is_file():
                try:
                    with info_path.open("r", encoding="utf-8") as ih:
                        info = json.load(ih)
                        info_parameters = info.get("parameters")
                        if info_parameters is not None:
                            # attach parameters either as "parameters" (overwriting) or as before
                            data["parameters"] = info_parameters
                except Exception:
                    # if info.json is broken, ignore and continue without parameters
                    pass

            # determine the aggregate's relative path *under* the dataset directory
            # this acts as our identifier for "same dataset item" across tasks
            try:
                rel_parent = agg_file.parent.relative_to(dataset_dir)
                rel_key = str(rel_parent)  # may be '.' for aggregate directly under dataset/
            except Exception:
                # fallback to using the agg filename's parent absolute path as key
                rel_key = str(agg_file.parent)

            # build the result dict and metadata
            result = dict(data)  # shallow copy so we can add metadata

            # keep only the latest by task_mtime for this rel_key
            existing = latest_per_relpath.get(rel_key)
            if existing is None or task_mtime > existing[0]:
                latest_per_relpath[rel_key] = (task_mtime, result)

    # return the kept results, sorted newest-first by mtime
    kept = [v[1] for v in sorted(latest_per_relpath.values(), key=lambda t: t[0], reverse=True)]
    return kept


