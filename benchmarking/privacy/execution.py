"""Top-level app-side privacy execution dispatch.

Acts as the single entry point for privacy jobs on the app side. It resolves the
requested or registered protocol, selects the appropriate runner, and handles
result persistence and cache-hit short-circuiting.
"""

from .job_models import PrivacyJobConfig, PrivacyProtocolId
from .membership_inference import run_membership_inference_job
from .model_inversion import run_model_inversion_job
from .metrics import PrivacyExecutionResult
from .persistence import privacy_result_exists, save_privacy_execution_result
from .property_inference import run_property_inference_job
from .reconstruction import run_reconstruction_job
from .protocol import resolve_requested_or_registered_protocol

_PROTOCOL_RUNNERS = {
    PrivacyProtocolId.MEMBERSHIP_INFERENCE: run_membership_inference_job,
    PrivacyProtocolId.MODEL_INVERSION: run_model_inversion_job,
    PrivacyProtocolId.PROPERTY_INFERENCE: run_property_inference_job,
    PrivacyProtocolId.RECONSTRUCTION: run_reconstruction_job,
}


def execute_privacy_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    """Execute one privacy job with optional cache-hit short-circuit and persistence."""
    protocol = resolve_requested_or_registered_protocol(job)
    runner = _PROTOCOL_RUNNERS.get(protocol)
    if runner is None:
        raise NotImplementedError(f"Privacy protocol '{protocol}' is not implemented.")

    if job.options.load_results and privacy_result_exists(job, protocol):
        from .persistence import load_privacy_execution_result
        return load_privacy_execution_result(job, protocol)

    if privacy_result_exists(job, protocol) and not job.options.overwrite:
        raise FileExistsError("Result exists. Set overwrite=true or load_results=true.")

    result = runner(job)
    save_privacy_execution_result(job, protocol, result)
    return result
