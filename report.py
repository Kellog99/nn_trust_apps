import argparse
import json
from pathlib import Path

from report import AdversarialReportGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Benchmark PDF generator.')

    parser.add_argument(
        "-d",
        '--benchmark_path',
        description="Path to the benchmark to print the report."
    )
    parser.add_argument("-m", '--model-dir', default=None)
    parser.add_argument("-o", '--output-path', default=None)
    args = parser.parse_args()

    GENERATION_AUTHORITY = "Ai Lab - Ldo"

    ########## 1) Getting the data ##########
    benchmark_path: Path = Path(args.benchmark_path).expanduser()
    if not benchmark_path.exists():
        raise FileNotFoundError("The benchmark path does not exist.")
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    ########## 2) Generating the report ##########

    report = AdversarialReportGenerator()
    report.generate(
        data=data,
        output_path=args.output_path,
        output_file_name=f"report_something.pdf",
        header_logo_path=None,
    )
