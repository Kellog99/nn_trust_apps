import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

from benchmarking.utils import evaluation
from nn_trust import ModelAdapter, StatisticComposer, Task


class AttackConfig(BaseModel):
    targeted: bool = False
    label_target: int = 0
    iou_threshold_evaluation: float = 0.5
    score_threshold_evaluation: float = 0.1


@pytest.fixture
def setup(monkeypatch):
    attack = SimpleNamespace(
        config=AttackConfig(),
        logger=Mock(),
        generate=Mock(side_effect=lambda x, y: x.clone()),
    )
    factory = Mock(return_value=attack)
    monkeypatch.setattr(evaluation.AttackFactory, "create", factory)
    model = ModelAdapter(torch.nn.Flatten())
    # Include a misclassified sample to guard against silently filtering inputs.
    inputs = torch.tensor([[[[0.9, 0.1]]], [[[0.8, 0.2]]], [[[0.1, 0.9]]]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    return model, loader, attack, factory


@pytest.mark.parametrize("attack_id", ["identitybaseline", "example"])
@pytest.mark.parametrize("limit", [None, 2])
def test_classification_metrics_artifacts_and_progress(setup, tmp_path, attack_id, limit):
    model, loader, attack, _ = setup
    statistics = StatisticComposer(statistics={"accuracy": {}})
    aggregate = Mock(wraps=statistics.update_aggregate)
    statistics.update_aggregate = aggregate
    result = evaluation.evaluate_attack(
        loader, model, statistics, attack_id,
        output_path=tmp_path, max_saved_elements=limit,
    )
    assert result.status == "finished"
    assert result.progress == result.total == 3
    assert result.result["accuracy"] == pytest.approx(2 / 3)
    assert aggregate.call_count == (attack_id != "identitybaseline")
    assert attack.generate.call_count == 2
    torch.testing.assert_close(
        attack.generate.call_args_list[0].kwargs["y"],
        torch.tensor([[-1., 0.], [0., -1.]]),
    )
    attack.logger.close.assert_called_once()
    artifacts = torch.load(tmp_path / attack_id / "log.pth", weights_only=False)
    assert len(artifacts["original_input"]) == (limit or 1)
    assert len(artifacts["adversarial_input"]) == (limit or 1)
    saved = json.loads((tmp_path / attack_id / "job_results.json").read_text())
    assert saved["status"] == "finished"
    assert saved["progress"] == 3


@pytest.mark.parametrize("failure", ["factory", "generate", "compute"])
def test_errors_are_persisted_and_resources_closed(setup, monkeypatch, tmp_path, failure):
    model, loader, attack, factory = setup
    logger = Mock()
    monkeypatch.setattr(evaluation, "PyTorchCheckpointLogger", Mock(return_value=logger))
    statistics = Mock()
    operation = {"factory": factory, "generate": attack.generate, "compute": statistics.compute}[failure]
    operation.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluation.evaluate_attack(loader, model, statistics, "example", output_path=tmp_path)
    saved = json.loads((tmp_path / "example" / "job_results.json").read_text())
    assert saved["status"] == "error"
    assert saved["error"] == "evaluation failed"
    assert logger.close.call_count == (failure != "factory")
    assert attack.logger.close.call_count == (failure != "factory")


@pytest.mark.parametrize("targeted", [False, True])
def test_detection_targets_and_single_inference_pair(setup, monkeypatch, tmp_path, targeted):
    _, _, attack, factory = setup
    attack.config.targeted = targeted
    output = (torch.zeros(1, 1, 4), torch.tensor([[[0.9, 0.1]]]))
    model = Mock(task=Task.Detection, return_value=output)
    prediction = {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]),
                  "scores": torch.tensor([0.9]), "cls_scores": output[1][0]}
    postprocess = Mock(return_value=[prediction])
    monkeypatch.setattr(evaluation, "nms", postprocess)
    inputs = torch.zeros(1, 3, 2, 2)
    labels = [{"boxes": torch.zeros(1, 4), "labels": torch.tensor([0])}]
    loader = DataLoader(list(zip(inputs, labels)), collate_fn=lambda items: tuple(zip(*items)))
    statistics = Mock()
    statistics.compute.return_value = {}
    evaluation.evaluate_attack(loader, model, statistics, "example", output_path=tmp_path)
    assert factory.call_args.kwargs["task"] == Task.Detection
    assert model.call_count == 2
    assert postprocess.call_count == 2
    target = statistics.update.call_args.kwargs["y_target"]
    assert target[0]["labels"].item() == int(targeted)
    statistics.update_aggregate.assert_called_once()


def test_advyolo_trains_then_evaluates_without_double_aggregation(setup, monkeypatch, tmp_path):
    model, loader, attack, _ = setup
    model.task = Task.Detection
    attack.generate.side_effect = None
    statistics = Mock()
    statistics.compute.return_value = {}
    frozen = Mock(return_value=statistics)
    monkeypatch.setattr(evaluation, "evaluate_frozen_advyolo", frozen)
    result = evaluation.evaluate_attack(
        loader, model, statistics, "advyoloevasion", output_path=tmp_path,
    )
    attack.generate.assert_called_once_with(gen_train=loader)
    frozen.assert_called_once()
    statistics.update_aggregate.assert_not_called()
    assert result.status == "finished"
    assert result.progress == result.total == 3
