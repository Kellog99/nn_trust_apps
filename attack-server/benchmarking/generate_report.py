"""
This script provide a simple cli interface
Starts with `python generate_report.py benchmark_run_dir`
Ten prompts the user for
1. Desired model
2. Desired dataset
Communicate output file location
"""
from pathlib import Path
import argparse
from benchmark_utils.pdf_report import create_benchmark_report

GENERATION_AUTHORITY = "Ai Lab - Ldo"

parser = argparse.ArgumentParser(prog='Benchmark PDF generator.')

parser.add_argument("-d", '--benchmark-dir', default=".")
parser.add_argument("-m", '--model-dir', default=None)
parser.add_argument("-o", '--output-path', default=None)
args = parser.parse_args()


if args.model_dir is None:
    output_path = args.output_path
    print(f"Please follow the prompts that will help you select dataset and models tuple to create PDF report.")
    benchmark_dir = Path(args.benchmark_dir).resolve()
    print(f"Available datasets.")
    datasets = [dir for dir in benchmark_dir.iterdir() if dir.is_dir()]
    for i, dir in enumerate(datasets):
        print(f"[{i}] {dir.name}")
    i = int(input("Select dataset: "))
    dataset_dir = datasets[i]

    print(f"Available models.")
    models = [dir for dir in dataset_dir.iterdir() if dir.is_dir()]
    for i, dir in enumerate(models):
        print(f"[{i}] {dir.name}")
    i = int(input("Select dataset: "))
    model_dir = models[i]

    output_path = input(f"choose a filename to save PDF report to: (ignored if blank): ")
    output_path = None if not output_path else output_path

    create_benchmark_report(
        benchmark_run_dir=str(benchmark_dir),
        model_name=model_dir.name,
        dataset_name=dataset_dir.name,
        filename=output_path,
        generated_by=GENERATION_AUTHORITY
    )
else:
    model_dir = Path(args.model_dir).resolve()
    print(f"Created PDF report for benchmark_run: {model_dir.parent.parent.parent.name}, dataset: {model_dir.parent.name}, model: {model_dir.name}")
    create_benchmark_report(
        benchmark_run_dir=str(model_dir.parent.parent),
        model_name=model_dir.name,
        dataset_name=model_dir.parent.name,
        filename=args.output_path,
        generated_by=GENERATION_AUTHORITY
    )