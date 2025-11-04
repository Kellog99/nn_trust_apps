from typing import List, Dict, Any, Iterable
from lib.models import BenchmarkModelProps
import numpy as np

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
    metric_keys = {k for item in sorted_data for k in item if k not in ['name', 'parameters']}
    
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
            metrics={k: item[k] for k in metric_keys if k in item}
            #metric_rank={k: int(rankings[k][idx]) for k in metric_keys if k in item}
        )
        for idx, item in enumerate(sorted_data)
    ]

def enrich_with_ranks(models, metrics_to_rank=None, ascending=False):
    """
    Enrich models with metric ranks.
    Args:
        models: List of model dictionaries with 'metrics' key
        metrics_to_rank: List of metric names to rank. If None, ranks all metrics.
        ascending: If True, lower values get better ranks. If False, higher values get better ranks.
    Returns:
        List of enriched models with ranks added to 'metrics' dict as '{metric_name}_rank'
    """
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