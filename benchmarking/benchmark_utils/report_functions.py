from pathlib import Path
import json
import logging as logger
from typing import List, Dict, Any, Optional, Iterable
from annotated_types import Gt, Ge, Le, Lt
from pydantic_core import PydanticUndefined
from pydantic import BaseModel
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF
from nn_trust.core import Task
import logging
import os
import numpy as np
# --------------- PYDANTIC MODELS --------------- #

class BenchmarkModelProps(BaseModel):
    name: str
    param: int
    task: str
    benchmark_id : str
    metrics: Dict[str, float | int]

class ParametersProps(BaseModel):
    name: str
    label: str
    min: float
    max: float
    step: float
    default: float
    description: str

class AttackProps(BaseModel):
    id: str
    name: str
    description: str
    knowledge: Optional[str] = None
    task: Optional[str] = None
    parameters: Optional[List[ParametersProps]] = None

# ----------------------------------------------- #

def get_attacks_info():
    """
    Get the list of all the available attacks with properties.
    """
    try:
        out = {}
        knowledge = {
            0: "White",
            1: "Black"
        }

        type = {
            0: "Physical",
            1: "Digital"
        }
        for atk in EAF.list_attacks(name=False):
            if hasattr(atk, "_name"):
                atk_id = atk.__name__.removesuffix("Attack").lower()
                # collecting all the parameters for displaying the configuration
                parameters = []
                for param_name, param_info in EAF.list_config_param(atk_id, (int, float)):
                    max_value = 1000
                    min_value = 1
                    if len(param_info.metadata) > 0:
                        # Extracting from the metadata the maximum value and minimum value of the parameters
                        for val in param_info.metadata:
                            if isinstance(val, (Gt, Ge)):
                                atr = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
                                if atr != -float('inf'):
                                    min_value = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
                                else:
                                    min_value = -10000
                            elif isinstance(val, (Lt, Le)):
                                atr = getattr(val, 'le' if isinstance(val, Le) else 'lt')
                                if atr != float('inf'):
                                    max_value = atr
                    if param_info.default is PydanticUndefined:
                        default = min_value
                    else:
                        default = param_info.default if param_info.default != float('inf') else max_value
                    if hasattr(param_info, 'step'):
                        step = getattr(param_info, 'step')
                    else:
                        step = (max_value - min_value) / 1000
                    parameters.append(
                        ParametersProps(
                            name=param_name,
                            label=param_name,
                            min=min_value,
                            max=max_value,
                            step=step,
                            default=default,
                            description=param_info.description
                        ))

                # Creating the list of all the attacks

                out[atk_id] = AttackProps(
                    id=atk_id,
                    name=atk._name,
                    knowledge=knowledge[atk.ATTACK_KNOWLEDGE.value],
                    description=atk._description if hasattr(atk,
                                                            "_description") else "this should be a description about this particular attack. However we have not being able to add that.",
                    type=type[atk.ATTACK_TYPE.value],
                    parameters=parameters
                )
            else:
                pass
        return out
    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")

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

def _coerce_number(x):
    """Convert string/value to int or float, otherwise return original value."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    try:
        s = str(x).replace(",", "")
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return x

def build_benchmark_dict(
    data: Iterable[Dict[str, Any]],
    dataset: str,
    task: str,
) -> Dict[str, Any]:
    """
    Build benchmark dictionary:
      - Sort by 'params' in ascending order
      - Group metrics into ordered lists
      - Ensure all lists have equal length (pad with None if needed)
    
    Returns dict like:
    {
      "dataset": ...,
      "task": ...,
      "accuracy": [...],
      "precision": [...],
      "f1score": [...],
      "robustness": [...],
      "params": [...]
    }
    """
    # Synonym mapping
    synonym_map = {
        "parameters": "params", "parameter": "params", "param": "params",
        "f1": "f1score", "f1_score": "f1score",
        "wobble": "wobbliness",
        "acc": "accuracy", "prec": "precision",
    }

    # Normalize keys
    normalized = []
    for entry in data:
        e = {}
        for k, v in entry.items():
            key_lower = k.strip().lower() if isinstance(k, str) else k
            canonical = synonym_map.get(key_lower, key_lower)
            e[canonical] = v
        
        # Coerce params to number
        if "params" in e:
            coerced = _coerce_number(e["params"])
            if isinstance(coerced, float) and coerced.is_integer():
                coerced = int(coerced)
            e["params"] = coerced
        
        normalized.append(e)

    # Sort by params (ascending)
    def _params_key(e):
        val = e.get("params")
        if val is None:
            return float("inf")  # Put entries without params at the end
        try:
            return float(val)
        except:
            return float("inf")
    
    sorted_entries = sorted(normalized, key=_params_key)

    # Collect all metric keys
    all_keys = set()
    for e in sorted_entries:
        all_keys.update(e.keys())
    
    # Preferred display order
    preferred = ["accuracy", "precision", "f1score", "robustness", "wobbliness", "params"]
    other = sorted([k for k in all_keys if k not in preferred])
    keys_ordered = [k for k in preferred if k in all_keys] + other

    # Build result
    result = {"dataset": dataset, "task": task}
    n = len(sorted_entries)

    for key in keys_ordered:
        values = []
        for e in sorted_entries:
            val = e.get(key)
            if val is not None:
                coerced = _coerce_number(val)
                # Convert metrics to float, keep params as int when possible
                if key != "params" and isinstance(coerced, (int, float)):
                    coerced = float(coerced)
                elif key == "params" and isinstance(coerced, float) and coerced.is_integer():
                    coerced = int(coerced)
                values.append(coerced)
            else:
                values.append(None)
        
        result[key] = values

    # Ensure all lists have equal length
    for key in keys_ordered:
        if len(result[key]) < n:
            result[key].extend([None] * (n - len(result[key])))
    
    return result

def transform_to_benchmark(raw_data: list[dict], task: str = 'image_classification') -> list[BenchmarkModelProps]:
    """Transform raw model data into sorted BenchmarkModelProps with metric rankings."""
    
    # Sort by parameters
    sorted_data = sorted(raw_data, key=lambda x: x['parameters'])
    
    # Extract metric keys (exclude 'name' and 'parameters')
    metric_keys = {k for item in sorted_data for k in item if k not in ['name', 'parameters', 'benchmark_id']}
    
    # Build metric matrix for vectorized ranking
    n_models = len(sorted_data)
    metric_matrix = {key: np.empty(n_models) for key in metric_keys}
    
    for idx, item in enumerate(sorted_data):
        for key in metric_keys:
            metric_matrix[key][idx] = item.get(key, -np.inf)
    
    # Calculate ranks (higher value = higher rank)
    #rankings = {key: np.argsort(np.argsort(-vals)) + 1 for key, vals in metric_matrix.items()}
    
    # Build result
    return [
        BenchmarkModelProps(
            name=item['name'],
            param=item['parameters'],
            task=task,
            benchmark_id=item['benchmark_id'],
            metrics={k: item[k] for k in metric_keys if k in item}
            #metric_rank={k: int(rankings[k][idx]) for k in metric_keys if k in item}
        ).model_dump()
        for idx, item in enumerate(sorted_data)
    ]

def enrich_with_ranks(models, metrics_to_rank=None, ascending=False):
    import copy
    # Deep copy to avoid modifying original data
    enriched_models = copy.deepcopy(models)
    
    # Get all metric names from first model if not specified
    if metrics_to_rank is None:
        metrics_to_rank = list(enriched_models[0]['metrics'].keys())
    
    # For each metric, compute ranks
    for metric_name in metrics_to_rank:
        # Extract metric values with model indices
        metric_values = [
            (i, model['metrics'][metric_name]) 
            for i, model in enumerate(enriched_models)
        ]
        
        # Sort by value (descending by default for "higher is better")
        metric_values.sort(key=lambda x: x[1], reverse=not ascending)
        
        # Assign ranks (1-based) directly in metrics dict
        for rank, (model_idx, value) in enumerate(metric_values, start=1):
            enriched_models[model_idx]['metrics'][f'{metric_name}_rank'] = rank
    
    return enriched_models

def extract_rank_metrics(models, model_name):
    """
    Given a list of model dictionaries and a model name, extract all metrics
    whose keys end with '_rank' for that model.

    Parameters:
        models (list): List of model dictionaries.
        model_name (str): Name of the model to search for.

    Returns:
        dict: A dictionary of metrics ending with '_rank', or an empty dict if not found.
    """
    for model in models:
        if model.get("name") == model_name:
            metrics = model.get("metrics", {})
            return {k: v for k, v in metrics.items() if k.endswith("_rank")}
    return {}    

