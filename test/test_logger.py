from pathlib import Path

import torch

from nn_trust.utils import PyTorchCheckpointLogger


def test_logged_artifacts(tmp_path: Path):
    logger = PyTorchCheckpointLogger(
        path=Path("./test_tmp/identitybaseline"),
        max_artifact={
            "original_input": 5,
            "adversarial_input": 5
        }
    )
    try:
        for _ in range(10):
            logger.log(
                tag="original_input",
                data=torch.rand(5, 5),
            )
            logger.log(
                tag="adversarial_input",
                data=torch.rand(5, 5),
            )
            logger.log(
                tag="other",
                data=torch.rand(5, 5),
            )
        logger.close()

        original_input = logger.get_log("original_input")
        adversarial_input = logger.get_log("adversarial_input")
        other = logger.get_log("other")
        print(other)
        assert len(original_input) == 5
        assert len(adversarial_input) == 5
        assert len(other) == 10
    finally:
        if logger._save_thread.is_alive():
            logger.close()
