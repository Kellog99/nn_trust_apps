from typing import List, Dict, Any, Iterable


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