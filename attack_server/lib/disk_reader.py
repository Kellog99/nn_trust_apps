import json
import logging as logger
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union


def find_model_and_task_dir(
        base_dir: Union[str, Path],
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        task_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Find the model directory and the task-id directory.

    Args:
        base_dir: root folder that contains 'benchmark_out' (e.g. '/path/to/NN_TRUST_APPS/attack_server')
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


def collect_dataset_aggregates_with_info(
        base_dir: str,
        dataset: str,
        *,
        filenames_pattern: str = "aggregate*.json",
        keep_latest_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Traverses base_dir for aggregate files for a given dataset.

    Features:
      - Logs traversal progress.
      - Optionally keeps only the latest aggregate (per relative path) based on task folder mtime.
      - Augments each aggregate dict with:
          "__task_dir": task folder name (string)
          "__task_mtime": task folder mtime (float)
          "__agg_path": full path to the aggregate file (string)
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"base_dir not found or not a directory: {base}")

    latest_per_relpath: Dict[str, list[tuple[float, Dict[str, Any]]]] = {}

    # iterate tasks in deterministic order
    for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        logger.info(f"Entering task directory: {task_dir.name} -> {task_dir}")
        try:
            task_mtime = float(task_dir.stat().st_mtime)
        except Exception:
            task_mtime = 0.0

        dataset_dir = task_dir / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            logger.info(f"  Dataset '{dataset}' not found in task {task_dir.name}, skipping.")
            continue

        logger.info(f"  Found dataset directory: {dataset_dir}")
        announced_model_dirs = set()

        for agg_file in dataset_dir.rglob(filenames_pattern):
            if not agg_file.is_file():
                continue

            model_dir = agg_file.parent
            model_dir_key = str(model_dir.relative_to(dataset_dir)) if model_dir != dataset_dir else "."
            if model_dir_key not in announced_model_dirs:
                logger.info(f"    Entering model directory: {model_dir_key} -> {model_dir}")
                announced_model_dirs.add(model_dir_key)

            logger.info(f"      Found aggregate file: {agg_file.name}")

            try:
                with agg_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                logger.warning(f"      WARNING: failed to read JSON from {agg_file}: {e} (skipping)")
                continue

            info_path = agg_file.parent / "info.json"
            if info_path.is_file():
                try:
                    with info_path.open("r", encoding="utf-8") as ih:
                        info = json.load(ih)
                        params = info.get("parameters")
                        m_name = info.get("name")
                        data["benchmark_id"] = str(task_dir).split(os.sep)[-1]
                        if params is not None:
                            data["parameters"] = params
                        if m_name is not None:
                            data["name"] = m_name
                            logger.info(f"        Loaded parameters from {info_path.name}")
                except Exception:
                    logger.warning(f"        WARNING: failed to read or parse {info_path.name} (ignoring)")

            try:
                rel_parent = agg_file.parent.relative_to(dataset_dir)
                rel_key = str(rel_parent)
            except Exception:
                rel_key = str(agg_file.parent)

            result = dict(data)
            latest_per_relpath.setdefault(rel_key, []).append((task_mtime, result))

    # --- Decide what to return ---
    if keep_latest_only:
        kept = [
            sorted(entries, key=lambda t: t[0], reverse=True)[0][1]
            for entries in latest_per_relpath.values()
        ]
        logger.info(f"Done. Collected {len(kept)} latest aggregates (per relative path).")
    else:
        kept = [
            result
            for entries in latest_per_relpath.values()
            for _, result in sorted(entries, key=lambda t: t[0], reverse=True)
        ]
        logger.info(f"Done. Collected {len(kept)} aggregates (all, not filtered).")

    return kept
