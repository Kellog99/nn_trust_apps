import base64
import io
import logging
from io import BytesIO
from typing import Any
from typing import Union

import numpy
import torch
import torchvision.transforms as transforms
import uvicorn
from PIL import Image
from fastapi import Body
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from nn_trust.attack import EvasionAttackFactory as EAF
from nn_trust.core import Task

from get_attacks import get_attacks
from get_metrics import get_metrics
from models import ReportProps, Error, generate_random_report, generate_benchmark_data, BenchmarkDataProps

app = FastAPI(name="mock-server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


################################## get functions ##################################
@app.get("/report/getResult", response_model=ReportProps, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_job_report_result(id: str) -> Union[ReportProps, Error]:
    """
    Get a TITANN benchmark report job result.
    """
    try:
        out = generate_random_report(id)
        return Response(
            status_code=200,
            content=out.model_dump_json()
        )

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())


@app.get("/benchmark/getResult", response_model=BenchmarkDataProps, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_job_benchmark_result(dataset: str, task: str) -> Union[BenchmarkDataProps, Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        out = generate_benchmark_data(dataset, task, 10)
        return Response(
            status_code=200,
            content=out.model_dump_json()
        )

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())


@app.get("/attacks/getInfo")
def get_attacks_info():
    """
    Get the list of all the available attacks for a specific task.
    """
    try:
        return get_attacks()

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")


@app.get("/metrics/getInfo")
def get_statistics_info():
    """
    Get the list of all the available attacks for a specific task.
    """
    try:
        return get_metrics()

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")


###################################################################################

################################## Post functions ##################################
@app.post("/attacks/executeAttack")
async def execute_attack(attack_info: Any = Body(None)):
    """
    This function executes the attack that is required from the frontend
    """
    try:
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(attack_info['image'])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Convert to tensor
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(image)
        pert = torch.randn_like(tensor_image).clamp(0, 1)
        adv = (tensor_image + pert).clamp(0, 1)
        # x = decode_image_to_tensor(attack_info.image)

        # Execute the attack and get results
        return {
            "x": attack_info['image'],
            "adv_perturbation": tensor_to_base64(pert),
            "x_adv": tensor_to_base64(adv),
            "original_prediction": "original",
            "adversarial_prediction": "adversarial",
            "confidence": [3 * numpy.sin(2 * numpy.pi * t / 100) + numpy.random.normal() for t in range(100)],
            "ssim": 1.3,
            "executionTime": 29.32
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack execution failed: {str(e)}")


def tensor_to_base64(tensor: torch.Tensor,
                     format: str = "PNG",
                     add_prefix: bool = True) -> str:
    """
    Encode a PyTorch image tensor (C×H×W, in [0,1] or [0,255]) into a valid Base64 string.
    Returns a RFC 4648-compliant string with proper padding.
    """
    tensor = tensor.detach().cpu()
    if tensor.max() <= 1:
        tensor = tensor * 255
    array = tensor.byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # Pad manually if needed (rare, but ensures validity)
    missing_padding = len(encoded) % 4
    if missing_padding:
        encoded += "=" * (4 - missing_padding)

    if add_prefix:
        encoded = f"data:image/{format.lower()};base64,{encoded}"

    return encoded


@app.post("/attacks/executeBenchmark")
async def execute_Benchmark(listAttacks: dict = Body(None)):
    """
    This function executes the benchmark.
    """
    try:
        print(listAttacks['banditprior']['parameters'])
        readable_status = {
            0: "Pending",
            1: "In Progress",
            2: "Completed",
            3: "Blocked"
        }
        # Execute the attack and get results

        out = [{
            "id": atk['id'],
            "name": atk['name'],
            "status": readable_status[numpy.random.randint(low=0, high=3)]
        } for key, atk in listAttacks.items()]
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


####################################################################################


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000
    )
