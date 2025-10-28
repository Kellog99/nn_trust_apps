try:
    from benchmark_utils.config import get_data_transformation_config
    from benchmark_utils.evaluator import Plan
    from benchmark_utils.utils import get_dataloader
except ModuleNotFoundError:
    # when used from attack.server or run as a subpackage use relative imports
    from .config import get_data_transformation_config
    from .evaluator import Plan
    from .utils import get_dataloader
import os
import inspect
from typing import Callable, Dict, List, Optional, Any
from abc import ABC, abstractmethod
import ray
from ray.util import ActorPool
from datetime import datetime
from fastapi.encoders import jsonable_encoder
# (get_dataloader imported above)


class Executor(ABC):
    @abstractmethod
    def execute_plan(self, plan: Plan):
        raise NotImplementedError


# ----------------- Utility for  dataset/model manipulation -----------------
def detect_node_data_root() -> Optional[str]:
    env = os.environ.get("ROOT_PATH")
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
    root = os.getenv("ROOT_PATH", "/home/cristiano-carta/Desktop/datasets")
    if os.path.isabs(path):
        p = path.split(os.getenv("ROOT_PATH")+os.sep)[1]
    else:
        p = path
    return os.path.join(root, p)   
    
@ray.remote
class ProgressTracker:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}

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
    
@ray.remote(num_gpus=0.5)
class SingleGPUActor:
    def __init__(self):
        try:
            self.node_root = self.detect_node_data_root()
        except Exception:
            self.node_root = os.getcwd()
        os.environ["RAY_OVERRIDE_ENVIRONMENT_VARIABLES_ALLOWLIST"] = "*"
        self.model = None
        self.dataloader = None

    def _ensure_dataset_root(self, dataset):
        #ds_root = dataset["source_path"]
        new_root = resolve_path(dataset["source_path"])
        #if ds_root != new_root:
        updated = update_dataset_root_obj(dataset, new_root)
        self.dataloader = updated
        #else:
        #    self.dataloader = update_dataset_root_obj(dataset,ds_root)

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
            actor_handle = SingleGPUActor.remote()
            actors.append(actor_handle)
        self.actors = actors
        # Create ActorPool from the actors
        self.pool = ActorPool(actors)
        return actors

    def execute_plan(self, plan: Plan) -> List[Any]:
        os.chdir("/home/cristiano-carta/Desktop/projects/nn_trust_apps/attack-server")
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
        task_ids = []
        for task_id, worker_conf in task_confs.items():
            worker_conf["tracker"] = self.tracker
            tasks.append((worker_action, worker_conf, plan.dataset, plan.model))
            task_ids.append(task_id)

        self.pool.map_unordered(
            lambda actor, args: actor.execute_worker.remote(*args),
            tasks
        )

        return None
