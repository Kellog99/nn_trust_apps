import random
from typing import Optional, Dict, List

from pydantic import BaseModel

######################################## Benchmark Validator ########################################
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
    knowledge: str
    type: str
    name: str
    parameters: List[ParametersProps]


######################################## Report Validator ########################################
class InfoProps(BaseModel):
    id: str
    name: str
    parameters: int
    classes: int
    dimensionality: List[int]


class MetricsProps(BaseModel):
    params: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    robustness: Optional[float] = None
    wobbliness: Optional[float] = None


class AttacksProps(BaseModel):
    name: str
    risk: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    misclassification: Optional[float] = None
    power: Optional[float] = None
    num_queries: Optional[int] = None
    robustness: Optional[List[float]] = None
    confusion_matrix: Optional[List[List[int]]] = None


class ReportProps(BaseModel):
    info: InfoProps
    metrics: MetricsProps
    attacks: Dict[str, AttacksProps]


##################################################################################################

class BenchmarkDataProps(BaseModel):
    dataset: str
    task: str
    accuracy: Optional[List[float]] = None
    precision: Optional[List[float]] = None
    f1score: Optional[List[float]] = None
    robustness: Optional[List[float]] = None
    wobbliness: Optional[List[float]] = None
    params: List[int]


class Error(BaseModel):
    code: int
    message: str


def generate_confusion_matrix(num_classes: int) -> List[List[int]]:
    """Generate a random confusion matrix for the given number of classes."""
    matrix = []
    for i in range(num_classes):
        row = []
        for j in range(num_classes):
            # Make diagonal elements (correct predictions) higher
            if i == j:
                row.append(random.randint(50, 200))
            else:
                row.append(random.randint(0, 30))
        matrix.append(row)
    return matrix


def generate_random_report(
        id: str,
        model_names: Optional[List[str]] = None,
        attack_names: Optional[List[str]] = None,
        num_attacks: int = 3,
        min_classes: int = 2,
        max_classes: int = 10,
        include_optional: bool = True
) -> ReportProps:
    """
    Generate a random ReportProps object.
    
    Args:
        model_names: List of possible model names. If None, uses default names.
        attack_names: List of possible attack names. If None, uses default names.
        num_attacks: Number of attacks to include in the report.
        min_classes: Minimum number of classes for the model.
        max_classes: Maximum number of classes for the model.
        include_optional: Whether to include optional fields with random values.
    
    Returns:
        ReportProps: A randomly generated report object.
    """
    if model_names is None:
        model_names = [
            "ResNet-50", "VGG-16", "AlexNet", "DenseNet-121", "MobileNet-v2",
            "EfficientNet-B0", "Inception-v3", "BERT-base", "GPT-3.5", "CLIP"
        ]

    if attack_names is None:
        attack_names = [
            "FGSM", "PGD", "C&W", "DeepFool", "AutoAttack", "JSMA",
            "Boundary", "HopSkipJump", "Square", "BanditsPriorRGF"
        ]

    # Generate InfoProps
    num_classes = random.randint(min_classes, max_classes)
    num_params = random.randint(1000000, 500000000)  # 1M to 500M parameters
    model_name = random.choice(model_names)

    info = InfoProps(
        id=id,  # Generate unique UUID
        name=model_name,
        parameters=num_params,
        classes=num_classes,
        dimensionality=[random.randint(32, 512) for _ in range(random.randint(2, 4))]
    )

    # Generate MetricsProps
    metrics = MetricsProps(
        params=num_params,
        accuracy=round(random.uniform(0.6, 0.98), 4) if include_optional else None,
        precision=round(random.uniform(0.6, 0.98), 4) if include_optional else None,
        f1score=round(random.uniform(0.6, 0.98), 4) if include_optional else None,
        confusion_matrix=generate_confusion_matrix(num_classes) if include_optional and random.choice(
            [True, False]) else None,
        robustness=round(random.uniform(0.1, 0.9), 4) if include_optional else None,
        wobbliness=round(random.uniform(0.0, 0.5), 4) if include_optional else None
    )

    # Generate AttacksProps
    attacks = {}
    selected_attacks = random.sample(attack_names, min(num_attacks, len(attack_names)))

    for attack_name in selected_attacks:
        attack = AttacksProps(
            name=attack_name,
            risk=round(random.uniform(0.0, 1.0), 4),
            accuracy=round(random.uniform(0.0, 0.8), 4) if include_optional and random.choice([True, False]) else None,
            precision=round(random.uniform(0.0, 0.8), 4) if include_optional and random.choice([True, False]) else None,
            f1score=round(random.uniform(0.0, 0.8), 4) if include_optional and random.choice([True, False]) else None,
            misclassification=round(random.uniform(0.0, 1.0), 4) if include_optional and random.choice(
                [True, False]) else None,
            power=round(random.uniform(0.0, 1.0), 4) if include_optional and random.choice([True, False]) else None,
            num_queries=random.randint(100, 10000) if include_optional and random.choice([True, False]) else None,
            robustness=[round(random.uniform(0.0, 1.0), 4) for _ in
                        range(random.randint(3, 8))] if include_optional and random.choice([True, False]) else None,
            confusion_matrix=generate_confusion_matrix(num_classes) if include_optional and random.choice(
                [True, False]) else None
        )
        attacks[attack_name.lower().replace("-", "_")] = attack

    return ReportProps(
        info=info,
        metrics=metrics,
        attacks=attacks
    )


def generate_benchmark_data(
        dataset: str,
        task: str,
        length: int,
        min_params: int = 1000000,
        max_params: int = 500000000,
        min_accuracy: float = 0.5,
        max_accuracy: float = 0.98,
        min_precision: float = 0.5,
        max_precision: float = 0.98,
        min_f1score: float = 0.5,
        max_f1score: float = 0.98,
        min_robustness: float = 0.0,
        max_robustness: float = 1.0,
        min_wobbliness: float = 0.0,
        max_wobbliness: float = 0.5,
        sort_by_params: bool = True,
        decimal_places: int = 4
) -> BenchmarkDataProps:
    """
    Generate a BenchmarkDataProps object with all lists populated.
    
    Args:
        length: Length of all lists (must be > 0)
        min_params: Minimum number of parameters
        max_params: Maximum number of parameters
        min_accuracy: Minimum accuracy value
        max_accuracy: Maximum accuracy value
        min_precision: Minimum precision value
        max_precision: Maximum precision value
        min_f1score: Minimum f1score value
        max_f1score: Maximum f1score value
        min_robustness: Minimum robustness value
        max_robustness: Maximum robustness value
        min_wobbliness: Minimum wobbliness value
        max_wobbliness: Maximum wobbliness value
        sort_by_params: Whether to sort all lists by parameter count (ascending)
        decimal_places: Number of decimal places for float values
        
    Returns:
        BenchmarkDataProps: A benchmark data object with all lists of specified length
        
    Raises:
        ValueError: If length is <= 0 or if min > max for any parameter
    """
    if length <= 0:
        raise ValueError("Length must be greater than 0")

    # Validate ranges
    if min_params >= max_params:
        raise ValueError("min_params must be less than max_params")
    if min_accuracy >= max_accuracy:
        raise ValueError("min_accuracy must be less than max_accuracy")
    if min_precision >= max_precision:
        raise ValueError("min_precision must be less than max_precision")
    if min_f1score >= max_f1score:
        raise ValueError("min_f1score must be less than max_f1score")
    if min_robustness >= max_robustness:
        raise ValueError("min_robustness must be less than max_robustness")
    if min_wobbliness >= max_wobbliness:
        raise ValueError("min_wobbliness must be less than max_wobbliness")

    # Generate parameter counts
    params = [random.randint(min_params, max_params) for _ in range(length)]

    # Generate metric lists
    accuracy = [round(random.uniform(min_accuracy, max_accuracy), decimal_places) for _ in range(length)]
    precision = [round(random.uniform(min_precision, max_precision), decimal_places) for _ in range(length)]
    f1score = [round(random.uniform(min_f1score, max_f1score), decimal_places) for _ in range(length)]
    robustness = [round(random.uniform(min_robustness, max_robustness), decimal_places) for _ in range(length)]
    wobbliness = [round(random.uniform(min_wobbliness, max_wobbliness), decimal_places) for _ in range(length)]

    # Sort all lists by parameter count if requested
    if sort_by_params:
        # Create list of tuples and sort by params
        combined = list(zip(params, accuracy, precision, f1score, robustness, wobbliness))
        combined.sort(key=lambda x: x[0])  # Sort by params (first element)

        # Unpack sorted data
        params, accuracy, precision, f1score, robustness, wobbliness = zip(*combined)
        params = list(params)
        accuracy = list(accuracy)
        precision = list(precision)
        f1score = list(f1score)
        robustness = list(robustness)
        wobbliness = list(wobbliness)

    return BenchmarkDataProps(
        dataset=dataset,
        task=task,
        accuracy=accuracy,
        precision=precision,
        f1score=f1score,
        robustness=robustness,
        wobbliness=wobbliness,
        params=params
    )
