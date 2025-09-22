import ray
import time
import asyncio
from typing import List, Any, Dict
from ray.util.queue import Queue
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Initialize Ray
ray.init(num_gpus=torch.cuda.device_count())

class SimpleImageClassifier(nn.Module):
    """Simple CNN for image classification (CIFAR-10 style)"""
    def __init__(self, num_classes=10):
        super(SimpleImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_random_batch(batch_size: int = 16) -> torch.Tensor:
    """Generate random image batch (3, 32, 32) for inference"""
    return torch.randn(batch_size, 3, 32, 32)

@ray.remote
class ProgressTracker:
    """Centralized progress tracking for all jobs"""
    def __init__(self, total_jobs: int):
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.job_results = {}
        self.job_status = {}
        
    def update_progress(self, job_id: str, status: str, progress: float = None, result: Any = None):
        self.job_status[job_id] = f"{status}: {progress:.1f}%" if progress else status
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

@ray.remote(num_gpus=0.25)
def pytorch_inference_worker(job_group: int, sub_job_id: int, num_batches: int, progress_tracker):
    """PyTorch inference worker that processes multiple batches"""
    job_id = f"job_{job_group}_sub_{sub_job_id}"
    
    try:
        # Initialize model and set to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleImageClassifier(num_classes=10)
        model.eval()
        model.to(device)
        
        # Initialize random weights (in real scenario, you'd load pre-trained weights)
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.randn_like(param.data) * 0.1
        
        print(f"🚀 {job_id}: Starting inference on {device} with {num_batches} batches")
        
        all_predictions = []
        all_confidences = []
        
        # Process batches with progress updates
        for batch_idx in range(num_batches):
            current_progress = ((batch_idx + 1) / num_batches) * 100
            
            # Update progress
            progress_tracker.update_progress.remote(
                job_id, "running", progress=current_progress
            )
            
            # Generate random batch (in real scenario, this would be real data)
            batch = create_random_batch(batch_size=16)
            batch = batch.to(device)
            
            # Perform inference
            with torch.no_grad():
                start_time = time.time()
                outputs = model(batch)
                inference_time = time.time() - start_time
                
                # Get predictions and confidence scores
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                max_confidences = torch.max(probabilities, dim=1)[0]
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_confidences.extend(max_confidences.cpu().numpy().tolist())
            
            # Simulate some processing time
            time.sleep(0.5)
        
        # Calculate statistics
        avg_confidence = np.mean(all_confidences)
        prediction_counts = np.bincount(all_predictions, minlength=10)
        most_common_class = np.argmax(prediction_counts)
        
        # Prepare result
        result = {
            "job_info": f"Group {job_group}, Sub-job {sub_job_id}",
            "total_samples": len(all_predictions),
            "avg_confidence": f"{avg_confidence:.3f}",
            "most_common_prediction": int(most_common_class),
            "class_distribution": prediction_counts.tolist(),
            "device_used": str(device),
            "batches_processed": num_batches
        }
        
        # Update status to completed
        progress_tracker.update_progress.remote(job_id, "completed", 100.0, result)
        
        print(f"✅ {job_id}: Completed inference - Avg confidence: {avg_confidence:.3f}")
        return result
        
    except Exception as e:
        # Update status to failed
        progress_tracker.update_progress.remote(job_id, "failed", 0, str(e))
        print(f"❌ {job_id}: Failed with error: {str(e)}")
        raise e

def launch_pytorch_inference_jobs(n_sub_jobs: int = 5, batch_range: tuple = (5, 15)):
    """
    Launch 2 job groups with PyTorch inference sub-jobs
    
    Args:
        n_sub_jobs: Number of sub-jobs per job group
        batch_range: Range for number of batches per job
    """
    total_jobs = 2 * n_sub_jobs
    progress_tracker = ProgressTracker.remote(total_jobs)
    
    # Create all sub-jobs for both job groups
    all_futures = []
    
    print(f"🔥 Launching PyTorch Inference Jobs")
    print(f"📊 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Job Group 1 - Image Classification Tasks
    print(f"🎯 Launching Job Group 1 (Classification) with {n_sub_jobs} sub-jobs...")
    for i in range(n_sub_jobs):
        num_batches = random.randint(*batch_range)
        future = pytorch_inference_worker.remote(1, i, num_batches, progress_tracker)
        all_futures.append(future)
    
    # Job Group 2 - More Classification Tasks
    print(f"🎯 Launching Job Group 2 (Classification) with {n_sub_jobs} sub-jobs...")
    for i in range(n_sub_jobs):
        num_batches = random.randint(*batch_range)
        future = pytorch_inference_worker.remote(2, i, num_batches, progress_tracker)
        all_futures.append(future)
    
    print(f"🚀 All {total_jobs} PyTorch inference jobs launched in parallel!")
    
    # Monitor progress
    monitor_progress(progress_tracker, total_jobs)
    
    # Wait for all jobs to complete
    results = ray.get(all_futures)
    
    # Get final progress report
    final_progress = ray.get(progress_tracker.get_progress.remote())
    
    print("\n" + "="*60)
    print("🎉 PYTORCH INFERENCE RESULTS:")
    print("="*60)
    
    total_samples = 0
    avg_confidences = []
    
    for job_id, result in final_progress["results"].items():
        if isinstance(result, dict):
            print(f"\n📋 {job_id}:")
            print(f"  • Samples processed: {result['total_samples']}")
            print(f"  • Average confidence: {result['avg_confidence']}")
            print(f"  • Most common class: {result['most_common_prediction']}")
            print(f"  • Device used: {result['device_used']}")
            print(f"  • Batches processed: {result['batches_processed']}")
            
            total_samples += result['total_samples']
            avg_confidences.append(float(result['avg_confidence']))
        else:
            print(f"❌ {job_id}: {result}")
    
    # Overall statistics
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  • Total samples processed: {total_samples}")
    print(f"  • Average confidence across all jobs: {np.mean(avg_confidences):.3f}")
    print(f"  • Successful jobs: {len([r for r in results if isinstance(r, dict)])}/{total_jobs}")
    
    return results, final_progress

def monitor_progress(progress_tracker, total_jobs, update_interval: float = 2.0):
    """Monitor and display progress in real-time"""
    print("\n" + "="*60)
    print("📊 MONITORING PYTORCH INFERENCE PROGRESS:")
    print("="*60)
    
    while True:
        progress = ray.get(progress_tracker.get_progress.remote())
        status_details = ray.get(progress_tracker.get_status.remote())
        
        # Display current status
        print(f"\n⏱️  Progress: {progress['completed']}/{total_jobs} "
              f"({progress['percentage']:.1f}%) completed")
        
        # Show individual job status
        for job_id, status in status_details.items():
            status_emoji = "✅" if "completed" in status else "🔄" if "running" in status else "❌"
            print(f"  {status_emoji} {job_id}: {status}")
        
        if progress['completed'] >= total_jobs:
            print("\n🎉 All PyTorch inference jobs completed!")
            break
            
        time.sleep(update_interval)


if __name__ == "__main__":
    print("🤖 Ray Cluster + PyTorch Inference Demo")
    print("="*60)
    
    # Check PyTorch setup
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU device: {torch.cuda.get_device_name(0)}")
    
    # PyTorch inference example
    print(f"\n🎯 PyTorch Inference Example: 2 job groups with 3 sub-jobs each")
    pytorch_results, pytorch_progress = launch_pytorch_inference_jobs(n_sub_jobs=3, batch_range=(8, 12))
    
    time.sleep(2)
    
    # Shutdown Ray
    ray.shutdown()
    print("\n🏁 PyTorch + Ray demo completed successfully!")