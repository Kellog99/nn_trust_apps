from pathlib import Path
from typing import Iterator

import ray
from tqdm import tqdm

from benchmarking.utils.evaluation import save_attack_result
from benchmarking.utils.execution import execute_job
from models import ModelReportProps


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
            root_path: str | Path = None,
            verbose: bool = False,
            use_ray: bool = False,
            num_gpus_per_job: float = 0.4
    ):
        self.root_path = Path(root_path).expanduser() if isinstance(root_path, str) else root_path
        self.verbose = verbose
        self.use_ray = use_ray
        self._remote_execute_job = ray.remote(num_gpus=num_gpus_per_job)(execute_job) if use_ray else None

    def save_results(self, job_output: dict):
        save_attack_result(
            benchmark_id=job_output["benchmark_job_info"]["benchmark_id"],
            atk_result=job_output["attack_results"],
            atk_id=job_output["benchmark_job_info"]["atk_id"],
            dataset_name=job_output["benchmark_job_info"]["dataset_id"],
            model_name=job_output["benchmark_job_info"]["model_id"],
            root_path=self.root_path,
        )

    def _iter_local(self, input_job_list) -> Iterator[ModelReportProps]:
        for job_config in tqdm(input_job_list, disable=not self.verbose):
            yield execute_job(job_config)

    def _iter_ray(self, input_job_list) -> Iterator[ModelReportProps]:
        pending = [self._remote_execute_job.remote(job_config) for job_config in input_job_list]
        with tqdm(total=len(pending), disable=not self.verbose) as pbar:
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                pbar.update(1)
                yield ray.get(done[0])

    def execute_jobs(self, input_job_list: list[dict]) -> dict:
        if not input_job_list:
            raise ValueError("input_job_list must not be empty")

        results_iter: Iterator[ModelReportProps] = self._iter_ray(input_job_list) if self.use_ray else self._iter_local(
            input_job_list)

        first_benchmark_id = None
        for job_result in results_iter:
            self.save_results(job_result)
            if first_benchmark_id is None:
                first_benchmark_id = job_result["benchmark_job_info"]["benchmark_id"]

        return {"output_path": Path(self.root_path) / first_benchmark_id}

    def __repr__(self):
        backend = "ray" if self.use_ray else "local"
        return f"{self.__class__.__name__}(root_path={self.root_path!r}, verbose={self.verbose!r}, backend={backend!r})"

    def __call__(self, input_job_list: list[dict]):
        if self.verbose:
            print(f"Starting execution of {len(input_job_list)} jobs via {self!r}")
        return self.execute_jobs(input_job_list)
