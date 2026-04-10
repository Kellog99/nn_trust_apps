"""
This file is responsible for
    1) Executing only the benchmark
    2) Creating the Report PDF
Whenever it is necessary
"""
import argparse
import json
from pathlib import Path

from report.pdf_sections.utils import ModelReportProps
from report.report import AdversarialReportStyle, AdversarialReportGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PDF reports or run benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'task',
        '--t',
        type=int,
        required=True,
        choices=[0, 1],
        help='0 for the Report PDF, 1 for the benchmark.'
    )
    ############################## Report variables ##############################

    parser.add_argument(
        "--report_path",
        "-rp",
        type=str,
        help="Path to the model's report position."
    )
    parser.add_argument(
        "--header_logo_path",
        "-hl",
        type=str,
        help="Path to the model's report position."
    )
    #################################################################################

    ############################## Benchmark variables ##############################
    parser.add_argument(
        'benchmark_config',
        '--bc',
        type=str,
        help="It represents the path to the benchmark config file.",
    )

    #################################################################################

    args = parser.parse_args()

    # Execute the appropriate command
    if args.task == 0:
        if not getattr(args, "report_path", None):
            raise ValueError("No report path provided.")

        report_path: Path = Path(getattr(args, "report_path")).expanduser()

        if report_path.exists() and report_path.is_file():
            with open(report_path, "r") as f:
                report = json.load(f)

            data: ModelReportProps = ModelReportProps.model_validate(report)
            data.info.repository = str(report_path.parent)
            adv_report_style = AdversarialReportStyle()
            adversarial_report = AdversarialReportGenerator(
                style=adv_report_style,
                excluded_metrics=["confusion_matrix"]
            )
            adversarial_report.generate(
                data=data,
                header_logo_path=getattr(args, "header_logo_path", None),
            )
        else:
            raise ValueError("The path is not valid. Please provide a valid path.")

    elif args.command == 'benchmark':
        pass
