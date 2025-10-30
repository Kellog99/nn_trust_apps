try:
    from benchmark_utils.config import get_data_transformation_config
    from benchmark_utils.evaluator import Plan
    from benchmark_utils.utils import get_dataloader
except ModuleNotFoundError:
    from .config import get_data_transformation_config
    from .evaluator import Plan
    from .utils import get_dataloader
import os
import inspect
from typing import Callable, Dict, List, Optional, Any
from abc import ABC, abstractmethod
import ray
import re
from ray.util import ActorPool
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import asyncio

class Executor(ABC):
    @abstractmethod
    def execute_plan(self, plan: Plan):
        raise NotImplementedError


# ----------------- Utility for  dataset/model manipulation -----------------
def detect_node_data_root() -> Optional[str]:
    env = os.environ.get("DATASET_REPO")
    if env:
        return env
    if os.path.exists("/data"):
        return "/data"
    return os.getcwd()

def update_dataset_root_obj(dataset: dict, new_root: str):

    transform, _ = get_data_transformation_config(
            transform_id=dataset["transform_config"]["transform_id"],
            size=dataset["transform_config"]["size"],
            crop=dataset["transform_config"].get("crop"),
            mean=dataset["transform_config"].get("mean"),
            std=dataset["transform_config"].get("std"),
        )

    return  get_dataloader(
            dataset=new_root,
            batch=dataset["batch"],
            subset=dataset["subset"],
            type_dataset=dataset["type_dataset"],
            transform=transform,
            num_workers=dataset["num_workers"],
            name=dataset["name"]
        )

def resolve_path(path: str) -> str:
    """
    Returns an absolute path. If the given path is not absolute,
    it prepends a root path read from the environment variable ROOT_PATH.
    """
    root = os.getenv("DATASETS_REPO",None)
    if not root:
        raise Exception("When performing parallel attacks, a DATASET_REPO env variable must be specified for every node!")
    if os.path.isabs(path):
        p = path.split(os.getenv("DATASET_REPO")+os.sep)[1]
    else:
        p = path
    return os.path.join(root, p)   

BENCHMARK_ID_REGEX = re.compile(r'^\d{8}T\d{6}$')  # adjust if your benchmark id format differs

def _iter_dirs(path: str):
    """Yield non-hidden directory names (sorted) inside path."""
    try:
        for name in sorted(os.listdir(path)):
            if name.startswith('.'):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                yield name
    except FileNotFoundError:
        return

def collect_benchmark_attacks(root: str) -> List[Dict[str, str]]:
    """
    Walks `root` (expected to be 'benchmarkids' folder) and collects every attack folder found.
    Returns list of {"benchmark_id": <id>, "attack": <attack_name>, "num_tasks": <count>}.
    Prints a simple folder-explorer view as it iterates.
    """
    root = os.path.abspath(root)
    results = []
    bench_to_attacks = {}

    print("Benchmark found on disk: ", os.path.basename(root) + os.sep)
    for bench in _iter_dirs(root):
        if not BENCHMARK_ID_REGEX.match(bench):
            continue
        bench_path = os.path.join(root, bench)
        print(f"  {bench}/")

        attacks_found = []
        for dataset in _iter_dirs(bench_path):
            ds_path = os.path.join(bench_path, dataset)
            print(f"    {dataset}/")
            for model in _iter_dirs(ds_path):
                model_path = os.path.join(ds_path, model)
                print(f"      {model}/")
                for attack in _iter_dirs(model_path):
                    print(f"        {attack}/")
                    attacks_found.append(attack)

        # store attacks found for this benchmark
        bench_to_attacks[bench] = attacks_found

    # build final results with num_tasks
    for bench, attacks in bench_to_attacks.items():
        num_tasks = len(attacks)
        for attack in attacks:
            results.append({
                "benchmark_id": bench,
                "attack": attack,
                "num_tasks": num_tasks
            })

    return results


@ray.remote
class ProgressTracker:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        on_disk_tasks = collect_benchmark_attacks(os.environ.get("BENCHMARK_OUTPUT_DIR","./benchmark_out"))
        for task in on_disk_tasks:
            self.create_task(task_id=f"{task["attack"]}_{task["benchmark_id"]}",
                             task_type="attack",
                             message="Task read from disk at boot up",
                             status="completed",
                             progress=100,
                             num_tasks=task["num_tasks"],
                             benchmark_id=task["benchmark_id"])

    def create_task(self, task_id: str, task_type: str, **kwargs) -> str:
        self.tasks[task_id] = {
            "task_type": task_type,
            "status": "created",
            "progress": 0,
            "message": "Task created",
            "created_at": jsonable_encoder(datetime.now()),
            "updated_at": jsonable_encoder(datetime.now()),
            "error": None,
            "result": None,
            **kwargs
        }
        return task_id

    def update_progress(self, task_id: str, status: str, progress: int = None, message: str = None, error: str = None, result: Any = None):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task["status"] = status
            task["updated_at"] = jsonable_encoder(datetime.now())
            if progress is not None:
                task["progress"] = progress
            if message is not None:
                task["message"] = message
            if error is not None:
                task["error"] = error
            if result is not None:
                task["result"] = result

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> Dict[str, Dict]:
        return self.tasks
    
    def wrapper(self, func, *args, **kwargs):
        """Wraps a callable to log before/after execution."""
        
        result = func(*args, **kwargs)

        return result

    def execute(self, func, *args, **kwargs):
        """Execute a callable with the wrapper."""
        return self.wrapper(func, *args, **kwargs)


@ray.remote(num_gpus=float(os.environ.get("FRACTION_FOR_GPU_ACTOR",1)))
class GPUActor:
    def __init__(self):
        try:
            self.node_root = self.detect_node_data_root()
        except Exception:
            self.node_root = os.getcwd()
        os.environ["RAY_OVERRIDE_ENVIRONMENT_VARIABLES_ALLOWLIST"] = "*"
        self.model = None
        self.dataloader = None

    def _ensure_dataset_root(self, dataset):
        new_root = resolve_path(dataset["source_path"])
        updated = update_dataset_root_obj(dataset, new_root)
        self.dataloader = updated

    def _ensure_model_state(self, model) -> None:
        self.model = model

    def execute_worker(self, 
                       worker_action: Callable, 
                       worker_conf: Dict[str, Any],
                       dataset : dict,   
                       model               
                        ):
        
        ds = dataset
        model = model
        self._ensure_dataset_root(ds)
        self._ensure_model_state(model)
        
        worker_conf["dataloader"] = self.dataloader
        worker_conf["model"] = self.model
         
        return worker_action(**worker_conf)





class RayActorPoolExecutor(Executor):
    def __init__(self,
                 num_actors: Optional[int] = None,
                ):
        self.num_actors = num_actors
        self.actors = []
        self.pool = None
        self.tracker = ProgressTracker.remote()

    def _create_actors(self, n: int) -> List[Any]:
        actors = []
        for _ in range(n):
            actor_handle = GPUActor.remote()
            actors.append(actor_handle)
        self.actors = actors
        self.pool = ActorPool(actors)
        return actors

    def execute_plan(self, plan: Plan, benchmark_id : str) -> List[Any]:
        initial_action = plan.action
        initial_params = plan.params
        initial_action(**initial_params)
        worker_action = plan.worker_action
        task_confs = plan.worker_params
        num_tasks = len(task_confs)
        if num_tasks == 0:
            return []
    
        num_actors = self.num_actors or min(num_tasks, len(task_confs))
        num_actors = max(1, min(num_actors, num_tasks))

        if not self.actors or len(self.actors) != num_actors:
            self._create_actors(num_actors)

        tasks = []
        for _, worker_conf in task_confs.items():
            worker_conf["tracker"] = self.tracker
            worker_conf["benchmark_id"] = benchmark_id
            worker_conf["num_tasks"] = num_tasks
            tasks.append((worker_action, worker_conf, plan.dataset, plan.model))

        async def launch_tasks():
            loop = asyncio.get_event_loop()
            
            loop.run_in_executor(None, lambda: list(self.pool.map_unordered(
                lambda actor, args: actor.execute_worker.remote(*args),
                tasks
            )))

        asyncio.create_task(launch_tasks())

        return benchmark_id
