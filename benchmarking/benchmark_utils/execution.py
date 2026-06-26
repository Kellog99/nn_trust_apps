import json
from pathlib import Path

import ray
import torch
from tqdm.auto import tqdm

from nn_trust.models.model_utils import load_model
from .dataset_utils import get_data_transformation_config, get_dataloader
from .evaluation_utils import evaluate_attack, save_attack_result_to_disk_v2


def override_keys_if_not_none(base_dict: dict, overriding_dict: dict) -> dict:
    """override keys in base dictionary using overriding dict if the latter are not none.
    Or the original dict do not contain the key to begin with"""
    res = dict(base_dict)  # create copy not to modify original element
    for k, v in overriding_dict.items():
        if v is not None or k not in res:
            res[k] = overriding_dict[k]
    return res


def execute_job(config: dict):
    """A function that use a full description of benchmark configuration and executor, is tasked to execute benchmark.
    """
    dataset = config["dataset"]
    model_config = config["model"]

    # Handle branching for NLP vs Image tasks
    if config.get("task_type") == "nlp":
        # NLP path: use NLPGoalDataset and skip image transforms
        dataloader = get_dataloader_nlp(
            dataset=dataset["source_path"],
            batch=dataset["batch"],
            name=dataset["name"]
        )
        dataset_default_config = {} # NLP datasets might not have this in the same way
    else:
        # Original Image-based path
        transform, inverse_transform = get_data_transformation_config(
            transform_id=dataset["transform_config"]["transform_id"],
            size=dataset["transform_config"].get("size"),
            crop=dataset["transform_config"].get("crop"),
            mean=dataset["transform_config"].get("mean"),
            std=dataset["transform_config"].get("std"),
        )
        dataloader = get_dataloader(
            dataset=dataset["source_path"],
            batch=dataset["batch"],
            subset=dataset["subset"],
            type_dataset=dataset["type_dataset"],
            transform=transform,
            num_workers=dataset["num_workers"],
            name=dataset["name"]
        )
        with open(Path(dataset["source_path"]) / (Path(dataset["source_path"]).name + ".json")) as f:
            # TODO: uniform this path to dataset_dir/info.json like for models instead of <dataset_dir>/<dataset_name>.json
            dataset_default_config = json.load(f)
    with open(Path(model_config["model_path"]) / "info.json") as f:
        model_default_config = json.load(f)

    model.metadata = override_keys_if_not_none(model_default_config, model_config)

    if config.get("task_type") == "nlp":
        dataloader.metadata = {} # Placeholder for now
        res = evaluate_attack_nlp(
            model=model,
            dataloader=dataloader,
            attack_config=config["attack"],
            statistics=config["evaluation"]["statistics"],
            device=torch.device("cuda") if (config["options"]["gpu"] and torch.cuda.is_available()) else torch.device("cpu")
        )
    else:
        dataloader.metadata = override_keys_if_not_none(dataset_default_config, dataset)
        res = evaluate_attack(
            model=model,
            dataloader=dataloader,
            attack_config=config["attack"],
            statistics=config["evaluation"]["statistics"],
            device=torch.device("cuda") if (config["options"]["gpu"] and torch.cuda.is_available()) else torch.device("cpu"),
            num_classes=dataset["num_classes"],
        )
    return {"attack_results": res, "benchmark_job_info": config["benchmark_info"]}


class LocalSerialExecutor:

    def __init__(self, root_path=None, verbose=False):
        self.root_path = root_path
        self.verbose = verbose

    def save_results(self, job_output):
        save_attack_result_to_disk_v2(
            benchmark_id=job_output["benchmark_job_info"]["benchmark_id"],
            atk_result=job_output["attack_results"],
            atk_id=job_output["benchmark_job_info"]["atk_id"],
            dataset_name=job_output["benchmark_job_info"]["dataset_id"],
            model_name=job_output["benchmark_job_info"]["model_id"],
            root_path=self.root_path
        )

    def execute_jobs(self, input_job_list: list[dict]):
        job_res = None
        for job_config in tqdm(input_job_list, disable=not self.verbose):
            results = execute_job(job_config)
            self.save_results(results)
            if job_res is None:
                job_res = results["benchmark_job_info"]["benchmark_id"]
        return {"output_path": Path(self.root_path) / job_res}

    def __repr__(self):
        return f"LocalExecutor(root_path='{self.root_path}', verbose='{self.verbose}')"

    def __call__(self, input_job_list: list):
        if self.verbose:
            print(f"Starting execution of {len(input_job_list)} jobs via LocalExecutor")
        return self.execute_jobs(input_job_list)


class LocalRayExecutor:

    def __init__(self, root_path=None, verbose=False):
        self.root_path = root_path
        self.verbose = verbose
        self.execute_job = ray.remote(num_gpus=0.4)(execute_job)

    def save_results(self, job_output):
        save_attack_result_to_disk_v2(
            benchmark_id=job_output["benchmark_job_info"]["benchmark_id"],
            atk_result=job_output["attack_results"],
            atk_id=job_output["benchmark_job_info"]["atk_id"],
            dataset_name=job_output["benchmark_job_info"]["dataset_id"],
            model_name=job_output["benchmark_job_info"]["model_id"],
            root_path=self.root_path
        )

    def execute_jobs(self, input_job_list: list[dict]):
        jobs_futures = [self.execute_job.remote(job_config) for job_config in input_job_list]
        jobs_results = ray.get(jobs_futures)
        for job_result in jobs_results:
            self.save_results(job_result)
        return {"output_path": Path(self.root_path) / jobs_results[0]["benchmark_job_info"]["benchmark_id"]}

    def __repr__(self):
        return f"LocalRayExecutor(root_path='{self.root_path}', verbose='{self.verbose}')"

    def __call__(self, input_job_list: list):
        if self.verbose:
            print(f"Starting execution of {len(input_job_list)} jobs via LocalExecutor")
        return self.execute_jobs(input_job_list)
