import logging
from typing import Union, List

import uvicorn
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import

from mocks import get_attacks
from models import ReportProps, Error, generate_random_report, generate_benchmark_data, BenchmarkDataProps, AttackProps

app = FastAPI(name="mock-server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


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


@app.get("/attacks/getInfo", response_model=List[AttackProps])
def get_attacks_info() -> Union[List[AttackProps], Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        out = get_attacks()
        return out

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error during get result")


if __name__ == "__main__":
    uvicorn.run(
        "app:app"
    )
