from utils import *

MODELS = [
    "timm/tf_efficientnet_b7.ap_in1k",  # Efficientnet trained with adversarial
    "timm/tf_efficientnet_b7.aa_in1k",
    "timm/convnextv2_base.fcmae_ft_in1k",
    "timm/convnextv2_huge.fcmae_ft_in1k",
    "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "timm/davit_base.msft_in1k"
]


def ray_search_hp():
    "This funciton actually execute the ray tune based HPO based on optuna"



    training_config = {
        "experiment_name": f"FT_304_{datetime.now().isoformat()}",
        "model_name": "timm/convnextv2_base.fcmae_ft_in1k", #ray.tune.choice(MODELS),
        "lr": ray.tune.loguniform(1e-5, 5e-4),
        "weight_decay": ray.tune.loguniform(1e-6, 1e-3),
        "batch_size": ray.tune.choice([16, 32, 64]),
        "epochs": 2,
        "data_dir": "/home/papab/data/aircraft-dataset/clean-aircraft-crop",  # earlier loaded
    }

    # train_model({
    #     "run_id": "test1",
    #     "model_name": "timm/tf_efficientnet_b7.ap_in1k",
    #     "lr": 1e-4,
    #     "weight_decay": 1e-6,
    #     "batch_size": 32,
    #     "epochs": 10,
    #     "data_dir": "/home/papab/data/aircraft-dataset/clean-aircraft-crop",  # earlier loaded
    # })


    optuna_search = OptunaSearch(metric="val_acc", mode="max")

    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(train_model),
            resources={"gpu": 1}
        ),
        param_space=training_config,
        tune_config=ray.tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=50,
        )
    )
    tuner.fit()


def train_one(model: str):
    "This funciton actually execute the ray tune based HPO based on optuna"


    train_model({
        "experiment_name": "304_model_training", #f"FT_304_train{datetime.now().isoformat()}",
        "model_name": model,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "batch_size": 32,
        "epochs": 30,
        "data_dir": "/home/papab/data/aircraft-dataset/clean-aircraft-crop",  # earlier loaded
    })




if __name__ == "__main__":
    for model in MODELS:
        print(f"Training model: {model}")
        train_one(model)