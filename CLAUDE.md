# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Setup

- **Dependency Management**: This project uses `uv`.
  - Sync environment: `uv sync --python 3.11`
  - Install local submodules: `uv pip install -e submodules/nn_trust/`
- **Git Submodules**: This project relies on Git submodules (e.g., `nn_trust`, `data_quality`).
  - Initialize: `git submodule init && git submodule update --recursive`

## Commonly Used Commands

- **Application Server (FastAPI)**:
  ```bash
  python app.py --reload --host 0.0.0.0 --port 8000
  ```
- **GUI (Next.js)**:
  ```bash
  cd frontend/
  yarn install
  yarn dev
  ```
- **Benchmarking**:
  ```bash
  python benchmark.py --MODELPATH <path> --DATASETPATH <path> --CONFIGPATH <path>
  ```
- **Generate Report**:
  ```bash
  python report.py --OUTPUTDIR <path_to_benchmark_output>
  ```

## Integration Updates (NLP)

- **Backend NLP Support**: Added `task_type` ("nlp" vs "classification") to `ExecutionConfig` for API/job dispatch.
- **Frontend NLP Support**: Frontend (`frontend`) now automatically detects model type (`llm` vs `cv`) to propagate `task_type` to the backend when executing attacks.

## High-Level Architecture

The repository is organized into three core components built on the `nn_trust` library:

1. **`attack_server/`**: The backend and job manager. It uses FastAPI for the API and Ray for job execution.
   - Entry point: `app.py`
   - Job management: `attack_server/routers/job_router.py`
   - Data access: `attack_server/lib/disk_reader.py`
2. **`frontend/`**: Next.js-based frontend.
3. **`benchmarking/`**: Orchestrates model evaluations.
   - Core logic: `benchmarking/benchmark_utils/`
   - Privacy metrics/attacks: `benchmarking/privacy/`
4. **`report/`**: PDF generation tools.
   - Section definitions: `report/pdf_sections/`

When making changes, prefer editing modules directly and verifying via the relevant execution script (e.g., `benchmark.py` for logic changes, `app.py` for API/server changes).
