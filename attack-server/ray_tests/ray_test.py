import ray
import time
import asyncio
from typing import List, Any, Dict
from ray.util.queue import Queue
import threading

# Initialize Ray
ray.init()

@ray.remote
class ProgressTracker:
    """Centralized progress tracking for all jobs"""
    def __init__(self, total_jobs: int):
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.job_results = {}
        self.job_status = {}
        
    def update_progress(self, job_id: str, status: str, progress : float = None, result: Any = None):
        self.job_status[job_id] = str(progress)
        #self.job_status[job_id]["progress"] = progress
        if status == "completed":
            self.completed_jobs += 1
            self.job_results[job_id] = result
        elif status == "failed":
            self.completed_jobs += 1
            self.job_results[job_id] = f"Failed: {result}"
    
    def get_progress(self) -> Dict:
        return {
            "completed": self.completed_jobs,
            "total": self.total_jobs,
            "percentage": (self.completed_jobs / self.total_jobs) * 100,
            "status": dict(self.job_status),
            "results": dict(self.job_results)
        }
    
    def get_status(self) -> Dict:
        return self.job_status
    
    def is_complete(self) -> bool:
        return self.completed_jobs >= self.total_jobs

@ray.remote
def sub_job_worker(job_group: int, sub_job_id: int, work_duration: float, progress_tracker):
    """Individual sub-job worker"""
    job_id = f"job_{job_group}_sub_{sub_job_id}"
    
    try:

        for i in range(10):
            current_progress = float((i + 1) / 100)
            # Update status to running
            progress_tracker.update_progress.remote(job_id, "running", progress = current_progress)
            
            # Simulate work
            time.sleep(work_duration)
        
        # Simulate some computation result
        result = f"Group {job_group}, Sub-job {sub_job_id} completed after {work_duration}s"
        
        # Update status to completed
        progress_tracker.update_progress.remote(job_id, "completed", result)
        
        return result
        
    except Exception as e:
        # Update status to failed
        progress_tracker.update_progress.remote(job_id, "failed", str(e))
        raise e

def launch_parallel_jobs(n_sub_jobs: int = 5, work_duration_range: tuple = (1, 3)):
    """
    Launch 2 jobs with n sub-jobs each, all running in parallel
    
    Args:
        n_sub_jobs: Number of sub-jobs per job group
        work_duration_range: Range for random work duration simulation
    """
    import random
    
    total_jobs = 2 * n_sub_jobs
    progress_tracker = ProgressTracker.remote(total_jobs)
    
    # Create all sub-jobs for both job groups
    all_futures = []
    
    # Job Group 1
    print(f"Launching Job Group 1 with {n_sub_jobs} sub-jobs...")
    for i in range(n_sub_jobs):
        duration = random.uniform(*work_duration_range)
        future = sub_job_worker.remote(1, i, duration, progress_tracker)
        all_futures.append(future)
    
    # Job Group 2
    print(f"Launching Job Group 2 with {n_sub_jobs} sub-jobs...")
    for i in range(n_sub_jobs):
        duration = random.uniform(*work_duration_range)
        future = sub_job_worker.remote(2, i, duration, progress_tracker)
        all_futures.append(future)
    
    print(f"All {total_jobs} sub-jobs launched in parallel!")
    
    # Monitor progress
    monitor_progress(progress_tracker, total_jobs)
    
    # Wait for all jobs to complete
    results = ray.get(all_futures)
    
    # Get final progress report
    final_progress = ray.get(progress_tracker.get_progress.remote())
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    for job_id, result in final_progress["results"].items():
        print(f"{job_id}: {result}")
    
    return results, final_progress

def monitor_progress(progress_tracker, total_jobs, update_interval: float = 1.0):
    """Monitor and display progress in real-time"""
    print("\n" + "="*50)
    print("MONITORING PROGRESS:")
    print("="*50)
    
    while True:
        progress = ray.get(progress_tracker.get_progress.remote())
        progress2 = ray.get(progress_tracker.get_status.remote())
        # Clear previous line and print progress
        #print(f"\rProgress: {progress['completed']}/{total_jobs} "
        #      f"({progress['percentage']:.1f}%) completed", end="", flush=True)
        print(progress2)
        if progress['completed'] >= total_jobs:
            print("\n✅ All jobs completed!")
            break
            
        time.sleep(update_interval)


if __name__ == "__main__":
    print("Ray Cluster Parallel Job Execution Demo")
    print("="*50)
    
    # Basic example
    print("\n1. Basic Example: 2 job groups with 3 sub-jobs each")
    basic_results, basic_progress = launch_parallel_jobs(n_sub_jobs=3)
    
    time.sleep(2)
    
    
    # Shutdown Ray
    ray.shutdown()
    print("\n🎉 Demo completed successfully!")