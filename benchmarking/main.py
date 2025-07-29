import json
from pathlib import Path
import os
import logging
import traceback

from utils.config import get_config, get_data_transformation_config, read_config_file
from pydantic import ValidationError
from utils.evaluator import Evaluator, EvaluatorConfig
from utils.utils import get_model, get_dataloader, get_structure, config_file_path_selector

if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    # get the parser
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    # benchmark_data = compose_benchmarking_data(config)

    for dataset in config["datasets"]:
        # for i, model_id in enumerate(config["model"]["list_models"]):
        for model_config in config["models"]:
            # transformations should depend on dataset and model
            try:
                transform, inverse_transform = get_data_transformation_config(
                    transform_id=dataset["transform_config"]["transform_id"],
                    size=dataset["transform_config"]["size"],
                    crop=dataset["transform_config"].get("crop"),
                    # mean=dataset["transform_config"].get("mean"),
                    # std=dataset["transform_config"].get("std"),
                )

                dataloader = get_dataloader(
                    dataset=dataset["source_path"],
                    batch=dataset["batch"],
                    subset=dataset["subset"],
                    type_dataset=dataset["type_dataset"],
                    transform=transform,
                    num_workers=dataset["num_workers"],
                )

                model = get_model(
                    model_name=model_config.get("name"),
                    model_type=model_config.get("type"),
                    model_weights_path=model_config.get("weights_path", None),
                    mean=dataset["transform_config"].get("mean"),
                    std=dataset["transform_config"].get("std"),
                    model_task=model_config.get("task")
                )

                output_path = Path(config["options"]["output_path"]) / dataset["name"]

                num_classes = (
                    model_config["num_classes"]
                    if model_config["num_classes"] > 0
                    else len(dataloader.dataset.dataset.classes)
                )

                evaluator = Evaluator(
                    config=EvaluatorConfig(
                        model=model,
                        dataloader=dataloader,
                        statistics=config["evaluation"]["statistics"],
                        statistic_average_method=config["evaluation"]["statistic_average_method"],
                        inverse_transformation=inverse_transform,
                        attacks=config["attacks"]["attack_list"],
                        attack_configurations=config["attacks"]["configurations"],
                        num_images_to_save=config["options"]["num_images_to_save"],
                        save_perturbation=config["options"]["save_perturbation"],
                        overwrite=config["options"]["overwrite"],
                        num_classes=num_classes,
                        output_path=output_path,
                        output_format=config["options"]["output_format"],
                    )
                )

                model_report = evaluator.evaluate()
                evaluator.save_results()
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()

        # Saving the structure for the report
        structure = get_structure(output_path)
        with open(output_path / "structure.json", "w") as f:
            json.dump(structure, f)
