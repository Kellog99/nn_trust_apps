"""Benchmarking package — imports guarded for environments where dependencies may be absent."""

import logging

try:
    from .main import benchmark, postprocess_benchmark_run_results
    from .benchmark_utils import (read_config_file, BenchmarkConfig,
                                  BenchmarkConfigModel, Evaluator,
                                  config_file_path_selector, get_model)
    from .benchmark_utils.report_functions import (
        transform_to_benchmark, get_attacks_info,
        collect_dataset_aggregates_with_info, BenchmarkModelProps,
        AttackProps, ParametersProps)
    from .benchmark_utils.pdf_report import AdversarialReportGenerator
    from .benchmark_utils.utils import get_dataloader
except (ModuleNotFoundError, ImportError):
    logging.warning("benchmarking package: evasion benchmarking unavailable — privacy-only mode.")
