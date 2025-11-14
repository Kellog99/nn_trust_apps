from pathlib import Path
import json
import logging as logger
from typing import List, Dict, Any, Optional, Union, Tuple
from annotated_types import Gt, Ge, Le, Lt
from pydantic_core import PydanticUndefined
from pydantic import BaseModel
from nn_trust.attack._evasion import EvasionAttackFactory as EAF
from nn_trust.attack.evaluation._statistics import StatisticsFactory as SF
from nn_trust.core import Task
import logging
import os

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

def get_attacks_info():
    """
    Get the list of all the available attacks for a specific task.
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


from typing import List, Dict, Any, Iterable
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


"""
Corporate Adversarial Attack Report Generator
A modular ReportLab-based PDF report generator for model robustness analysis
"""

import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class CorporateColors:
    """Corporate color palette"""
    BLACK = colors.HexColor('#000000')
    DARK_GRAY = colors.HexColor('#333333')
    GRAY = colors.HexColor('#666666')
    LIGHT_GRAY = colors.HexColor('#CCCCCC')
    VERY_LIGHT_GRAY = colors.HexColor('#F5F5F5')
    RED = colors.HexColor('#CC0000')
    WHITE = colors.HexColor('#FFFFFF')


class ReportStyles:
    """Centralized style definitions"""
    
    @staticmethod
    def get_styles():
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=28,
            textColor=CorporateColors.BLACK,
            spaceAfter=30,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 1
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=CorporateColors.RED,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=CorporateColors.RED,
            borderPadding=5,
            leftIndent=0
        ))
        
        # Heading 2
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=CorporateColors.DARK_GRAY,
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=CorporateColors.DARK_GRAY,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Metric label
        styles.add(ParagraphStyle(
            name='MetricLabel',
            fontSize=9,
            textColor=CorporateColors.GRAY,
            fontName='Helvetica-Bold'
        ))
        
        # Metric value
        styles.add(ParagraphStyle(
            name='MetricValue',
            fontSize=11,
            textColor=CorporateColors.BLACK,
            fontName='Helvetica'
        ))
        
        return styles


class HeaderFooter:
    """Page header and footer handler"""
    
    def __init__(self, logo_path=None):
        self.logo_path = logo_path
    
    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()
        
        # Header
        if self.logo_path:
            try:
                canvas_obj.drawImage(
                    self.logo_path,
                    40, A4[1] - 80,
                    width=100, height=100,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            except:
                pass  # Skip if logo not found
        
        # Red header line
        canvas_obj.setStrokeColor(CorporateColors.RED)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(40, A4[1] - 60, A4[0] - 40, A4[1] - 60)
        
        # Footer
        canvas_obj.setStrokeColor(CorporateColors.LIGHT_GRAY)
        canvas_obj.setLineWidth(1)
        canvas_obj.line(40, 40, A4[0] - 40, 40)
        
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(CorporateColors.GRAY)
        canvas_obj.drawString(40, 25, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas_obj.drawRightString(A4[0] - 40, 25, f"Page {doc.page}")
        
        canvas_obj.restoreState()


class ModelInfoSection:
    """Model information section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        info = data.get('info', {})
        
        title = Paragraph("Model Information", styles['CustomHeading1'])
        
        # Model info table
        table_data = [
            ['Model Name', info.get('name', 'N/A')],
            ['Parameters', f"{info.get('parameters', 0):,}"],
            ['Classes', str(info.get('classes', 'N/A'))],
            ['Input Dimensions', str(info.get('dimensionality', 'N/A'))]
        ]
        
        table = Table(table_data, colWidths=[150, 350])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
            ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
            ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        # Keep title and table together
        elements.append(KeepTogether([title, table]))
        elements.append(Spacer(1, 20))
        
        return elements


class MetricsSection:
    """Global metrics section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        metrics = data.get('metrics', {})
        
        if not metrics:
            return elements
        
        title = Paragraph("Global Metrics", styles['CustomHeading1'])
        
        # Metrics table
        table_data = []
        for key, value in metrics.items():
            formatted_value = MetricsSection._format_metric(value)
            table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        if table_data:
            table = Table(table_data, colWidths=[150, 350])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
                ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
                ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            # Keep title and table together
            elements.append(KeepTogether([title, table]))
            elements.append(Spacer(1, 20))
        
        return elements
    
    @staticmethod
    def _format_metric(value):
        if isinstance(value, float):
            return f"{value:.4f}"  
        elif isinstance(value, int):
            return f"{value:,}"
        return str(value)


class AttacksSection:
    """Adversarial attacks section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        attacks = data.get('attacks', {})
        
        if not attacks:
            return elements
        
        elements.append(Paragraph("Adversarial Attacks Analysis", styles['CustomHeading1']))
        
        # Summary table
        elements.extend(AttacksSection._build_summary_table(attacks, styles))
        
        # Detailed attack reports
        for attack_name, attack_data in attacks.items():
            #if attack_name != 'reference':
                elements.extend(AttacksSection._build_attack_detail(
                    attack_name, attack_data, styles
                ))
        
        return elements
    
    @staticmethod
    def _build_summary_table(attacks, styles):
        elements = []
        title = Paragraph("Attack Summary", styles['CustomHeading2'])
        
        # Prepare summary data
        headers = ['Attack', 'Robustness', 'Accuracy', 'SSIM', 'Misclass.']
        table_data = [headers]
        
        for attack_name, attack_data in attacks.items():
            #if attack_name != 'reference':
                row = [
                    attack_name.upper(),
                    AttacksSection._format_value(attack_data.get('robustness')),
                    AttacksSection._format_value(attack_data.get('accuracy')),
                    AttacksSection._format_value(attack_data.get('ssim')),
                    AttacksSection._format_value(attack_data.get('misclassification'))
                ]
                table_data.append(row)
        
        table = Table(table_data, colWidths=[80, 100, 80, 80, 80])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), CorporateColors.RED),
            ('TEXTCOLOR', (0, 0), (-1, 0), CorporateColors.WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), CorporateColors.WHITE),
            ('TEXTCOLOR', (0, 1), (-1, -1), CorporateColors.DARK_GRAY),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [CorporateColors.WHITE, CorporateColors.VERY_LIGHT_GRAY]),
            
            # General styling
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        # Keep title and table together
        elements.append(KeepTogether([title, table]))
        elements.append(Spacer(1, 20))
        
        return elements
    
    @staticmethod
    def _build_attack_detail(attack_name, attack_data, styles):
        elements = []
        
        title = Paragraph(f"<b>{attack_name.upper()}</b> Attack Details", 
                         styles['CustomHeading2'])
        
        # Metrics table
        table_data = []
        metric_order = [
            'countsamples', 'accuracy', 'precision', 'robustness',
            'misclassification', 'f1score', 'expectedcalibrationerror',
            'meansquarecontingency', 'ssim'
        ]
        
        for metric in metric_order:
            if metric in attack_data:
                label = metric.replace('_', ' ').title()
                value = AttacksSection._format_value(attack_data[metric])
                table_data.append([label, value])
        
        if table_data:
            table = Table(table_data, colWidths=[200, 250])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
                ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
                ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            # Keep title and table together
            elements.append(KeepTogether([title, table]))
            elements.append(Spacer(1, 15))
        
        return elements
    
    @staticmethod
    def _format_value(value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f"{value:.4f}"  
        elif isinstance(value, int):
            return f"{value:,}"
        return str(value)

class AdversarialReportGenerator:
    """Main report generator class"""
    
    def __init__(self, logo_path=None):
        """
        Initialize report generator
        
        Args:
            logo_path: Path to corporate logo PNG file
        """
        self.logo_path = logo_path
        self.styles = ReportStyles.get_styles()
    
    def generate(self, data, output_path='adversarial_report.pdf'):
        """
        Generate PDF report from JSON data
        
        Args:
            data: Dictionary with model and attack information
            output_path: Output PDF file path
        """
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=80,
            bottomMargin=60
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._build_title_page(data))
        
        # Model information
        story.extend(ModelInfoSection.build(data, self.styles))
        
        # Global metrics
        story.extend(MetricsSection.build(data, self.styles))
        
        # Attacks analysis
        story.extend(AttacksSection.build(data, self.styles))
        
        # Build PDF
        doc.build(
            story,
            onFirstPage=HeaderFooter(self.logo_path),
            onLaterPages=HeaderFooter(self.logo_path)
        )
        
        print(f"Report generated: {output_path}")
    
    def _build_title_page(self, data):
        elements = []
        
        # Title
        title = Paragraph(
            "Model Trustworthiness Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        
        # Subtitle with model name
        model_name = data.get('info', {}).get('name', 'Unknown Model')
        subtitle = Paragraph(
            f"<font color='#{CorporateColors.RED.hexval()[2:]}'>Model: {model_name}</font>",
            self.styles['CustomHeading2']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 10))
        
        # Report metadata
        metadata_text = f"""
        <font size=10>
        <b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Generated By:</b> Leonardo S.p.A. <br/>
        </font>
        """
        elements.append(Paragraph(metadata_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 30))
        
        # Executive summary
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading2']))
        
        summary_text = f"""
        This report provides a comprehensive analysis of the adversarial robustness 
        of the {model_name} model. The analysis includes multiple attack and evaluates the model's resilience against adversarial perturbations.
        """
        elements.append(Paragraph(summary_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 20))
        
        return elements
