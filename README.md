# NN Trust Applications

Backend and command-line tools for running adversarial-robustness benchmarks with the bundled [`nn_trust`](submodules/nn_trust) library. A run evaluates one or more models against a dataset, saves per-attack results, and can generate a PDF report.

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git submodule update --init --recursive
uv sync --python 3.11
```

## Run a benchmark

Models and datasets are local directories with an `info.json` file. Point the CLI configuration at those directories:

```yaml
model:
  - source_path: /path/to/model
datasets:
  - source_path: /path/to/dataset
attacks:
  - id: fgsm
metrics:
  - id: accuracy
options:
  output_path: ./benchmark_out
  subset: 100
  gpu: false
```

Run it with:

```bash
uv run python benchmark.py --config_path path/to/config.yaml
```

The identity baseline is always included. Results are written to:

```text
<output_path>/<benchmark_id>/<model_id>/<dataset_id>/
├── report.json
└── <attack_id>/job_results.json
```

Create a PDF from a completed report:

```bash
uv run python report.py --benchmark_path path/to/report.json --output_path ./reports
```

## Security report PDF

The default report follows the frontend dashboard on white A4 pages: model information,
preprocessing, metric cards, benchmarking, and risk-based vulnerability assessment.
The generator accepts `ModelReportProps` or a report dictionary. Dictionaries preserve
frontend field order and additional fields such as attack `category`.

Every page displays `report/images/Logo_Leonardo.png` in the top-left header,
within a 140 × 28 pt box preserving its proportions. A minimum 68 pt top margin
keeps content clear of the logo. Pass `header_logo_path` to `generate()` to use
a different logo; the bundled image is used by default, including CLI/API calls.

```python
from report import AdversarialReportGenerator

generator = AdversarialReportGenerator(
    benchmark=[
        {"name": "Reference model", "param": 25_000_000,
         "metrics": {"accuracy": 0.91}},
    ],
    include_attack_details=True,
)
generator.generate(report_data, output_path="out/security-report.pdf")
```

`benchmark` accepts the objects returned by `/report/benchmarks`, a mapping of model
names to those objects, or the legacy metric-to-score-list format. Legacy scores
have no parameter counts, so they appear in the leaderboard only. Accuracy is the
preferred comparison metric; otherwise the first finite scalar metric is selected.
Without supplied benchmarks, the PDF explicitly displays an empty state. The PDF
generator does not fetch benchmarks; existing callers must pass them to enable the
comparison chart.

Individual attack metrics and parameters are included by default, including through
`report.py`. Set `include_attack_details=False` only for a summary-only report.
Metrics whose names start with `Class` (case-insensitive, including `classrobustness`)
are rendered as bar charts in both global metrics and attack details. Lists use
zero-based class IDs on the X axis; dictionaries use their keys as class labels.
Each metric has a single chart containing all classes, even above 24 classes; X-axis
tick labels are hidden. Missing or non-finite values are marked `N/A`, not plotted
as zero.

Confusion matrices are shown as heatmaps using the original values, both globally
and for individual attacks. Each attack starts on a new page; parameters use compact
cards, preserving their values without metric rounding. Long metric arrays use
splittable rows so they can continue across pages. Input data is never modified.

Use `AdversarialReportStyle(pagesize=landscape(A4))` for landscape output. Both
orientations stack the chart and leaderboard to allow long rankings to paginate.

Examples are loaded from `<report repository>/<attack ID>/log.pth` (tensor-only
loading) or legacy `<sample>_original.png`, `_pert.png`, `_adv.png` images (JPEG is
also supported). The top-level `repository` provided by `repository_router.py` is
used automatically. For other callers, pass `examples_root` to `generate()`; the
CLI uses the directory containing `report.json`. Tensor images are denormalized
with the report's preprocessing mean/std when compatible; perturbations are shown
as absolute differences scaled for visibility. Missing examples are indicated in
the report. Original artifacts and metrics are never changed.

Run the focused PDF tests with:

```bash
uv run pytest test/test_report_pdf.py -q
```

## API

Start the FastAPI server:

```bash
uv run python app.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The main benchmark endpoints are `POST /job/start_benchmark`, `GET /job/getJobs`, and `GET /job/getReport`. Server paths and other settings can be supplied as CLI options or in a JSON file passed with `--configuration_file`.

For a complete API request example and CIFAR-10 helper commands, see [benchmarking/README.md](benchmarking/README.md).
