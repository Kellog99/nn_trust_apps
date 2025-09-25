from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from models import ReportProps, Error, generate_random_report, generate_benchmark_data, BenchmarkDataProps, AttackProps
from typing import Union, List
from mocks import get_attacks
import logging
import uvicorn

app = FastAPI(name="mock-server")


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
    

@app.get("/attacks/getInfo", response_model=List[AttackProps], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_attacks_info() -> Union[BenchmarkDataProps, Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        out = get_attacks()
        return JSONResponse(
            status_code=200,
            content=[o.model_dump_json() for o in out]
        )

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app"
    )