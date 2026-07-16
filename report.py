import argparse
import json
from pathlib import Path

from models import ModelReportProps
from report import AdversarialReportGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Benchmark PDF generator.')

    parser.add_argument(
        '--benchmark_path',
        help="Path to the benchmark to print the report."
    )
    parser.add_argument(
        '--output_path',
        default="./out",
        type=str,
        help="Path to the output report."
    )

    args = parser.parse_args()

    GENERATION_AUTHORITY = "Ai Lab - Ldo"

    ########## 1) Getting the data ##########
    benchmark_path: Path = Path(args.benchmark_path).expanduser()
    if not benchmark_path.is_file():
        benchmark_path = benchmark_path / "report.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("The benchmark path does not exist.")
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    ########## 2) Checking whether the data are correct ##########
    data = ModelReportProps.model_validate(data)

    ########## 3) Generating the report ##########
    file_name: str = data.info.name or data.info.id or "model_adversarial_report.pdf"
    file_name = file_name.replace(" ", "_").lower()
    # Adding the proper extension to the file
    if not file_name.endswith(".pdf"):
        file_name = file_name + ".pdf"

    output_path: Path = Path(getattr(data, "output_path", args.output_path)).expanduser() / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = AdversarialReportGenerator()
    report.generate(
        data=data,
        output_path=output_path,
        header_logo_path=None,
    )
