from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ray
import uuid
import time
from typing import Dict, Any, Optional
from enum import Enum

# Initialize Ray
if not ray.is_initialized():
    ray.init()

app = FastAPI()

# Global dictionary to store task references
# In production, use Redis or a database
TASK_STORE: Dict[str, ray.ObjectRef] = {}

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None

class TaskStartResponse(BaseModel):
    task_id: str
    status: str = "PENDING"

# Example Ray remote function
@ray.remote
def long_running_task(duration: int, task_name: str):
    """Simulate a long-running task"""
    print(f"Starting task: {task_name}")
    
    # Simulate work
    for i in range(duration):
        time.sleep(10)
        print(f"Task {task_name}: Step {i+1}/{duration}")
    
    return {
        "task_name": task_name,
        "duration": duration,
        "completed_at": time.time(),
        "result": f"Task {task_name} completed successfully!"
    }

@ray.remote
def data_processing_task(data: list):
    """Example data processing task"""
    time.sleep(2)  # Simulate processing time
    
    # Process data
    processed = [x * 2 for x in data]
    total = sum(processed)
    
    return {
        "original_data": data,
        "processed_data": processed,
        "total": total,
        "count": len(data)
    }

@app.post("/tasks/start", response_model=TaskStartResponse)
async def start_task(duration: int = 5, task_name: str = "default"):
    """Start a long-running task"""
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Start the Ray task
    object_ref = long_running_task.remote(duration, task_name)
    
    # Store the object reference
    TASK_STORE[task_id] = object_ref
    
    return TaskStartResponse(task_id=task_id)

@app.post("/tasks/process-data", response_model=TaskStartResponse)
async def start_data_processing(data: list):
    """Start a data processing task"""
    
    task_id = str(uuid.uuid4())
    object_ref = data_processing_task.remote(data)
    TASK_STORE[task_id] = object_ref
    
    return TaskStartResponse(task_id=task_id)

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Monitor task progress and get result"""
    
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    
    object_ref = TASK_STORE[task_id]
    
    try:
        # Check if task is ready (non-blocking)
        ready_refs, remaining_refs = ray.wait([object_ref], timeout=0)
        
        if ready_refs:
            # Task is complete, get the result
            try:
                result = ray.get(ready_refs[0])
                # Clean up completed task
                del TASK_STORE[task_id]
                
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.SUCCESS,
                    result=result
                )
            except Exception as e:
                # Task failed
                del TASK_STORE[task_id]
                return TaskResponse(
                    task_id=task_id,
                    status=TaskStatus.FAILURE,
                    error=str(e)
                )
        else:
            # Task is still running
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking task: {str(e)}")

@app.get("/tasks/{task_id}/wait")
async def wait_for_task(task_id: str, timeout: int = 30):
    """Wait for task completion with timeout"""
    
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    
    object_ref = TASK_STORE[task_id]
    
    try:
        # Wait for task completion with timeout
        ready_refs, remaining_refs = ray.wait([object_ref], timeout=timeout)
        
        if ready_refs:
            result = ray.get(ready_refs[0])
            del TASK_STORE[task_id]
            
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                result=result
            )
        else:
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
            
    except Exception as e:
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILURE,
            error=str(e)
        )

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    
    object_ref = TASK_STORE[task_id]
    
    try:
        # Cancel the task
        ray.cancel(object_ref)
        del TASK_STORE[task_id]
        
        return {"message": f"Task {task_id} cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

@app.get("/tasks")
async def list_tasks():
    """List all active tasks"""
    active_tasks = []
    
    for task_id, object_ref in TASK_STORE.items():
        ready_refs, remaining_refs = ray.wait([object_ref], timeout=0)
        status = "SUCCESS" if ready_refs else "PENDING"
        
        active_tasks.append({
            "task_id": task_id,
            "status": status
        })
    
    return {"active_tasks": active_tasks}

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on app shutdown"""
    if ray.is_initialized():
        ray.shutdown()

# Example usage endpoints for testing
@app.get("/")
async def root():
    return {
        "message": "FastAPI + Ray Task Monitoring",
        "endpoints": {
            "start_task": "POST /tasks/start?duration=10&task_name=test",
            "process_data": "POST /tasks/process-data with JSON body: [1,2,3,4,5]",
            "check_status": "GET /tasks/{task_id}",
            "wait_for_task": "GET /tasks/{task_id}/wait?timeout=30",
            "cancel_task": "DELETE /tasks/{task_id}",
            "list_tasks": "GET /tasks"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)