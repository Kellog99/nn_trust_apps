# Benchmarking
In this part it will be explained how to perform a `benchmark` for a specific model.

There are several way to import and benchmark a model: 
* From `timm` library as a quick reference (if you have internet access);
* From a local model saved as a `.pth` file with weights and model;
* From a local model saved as weights in a file `.pt` and the inference of the model specified in a Python3 file.

To run the benchmarking routine, run
```commandline
python main.py
```
with the configurations stored in `config` as YAML.

### Configuration File
The configuration file is a `YAML` file. Exists a default configuration file `scripts/config.yaml` which contains all
the most important variables for the configuration files. The structure of this file is as follows.
```yaml
evaluation:
  statistics: # statistics is a list of dictionaries
    - name: 'countsamples' # counts the number of samples
    - name: 'accuracy' 
      average: 'macro'
    - name: 'precision'
      average: 'macro'
    - name: 'robustness'
    - name: 'misclassification'

options:
  load_results: false                 # loading any existing results.
  overwrite: true                     # overwriting the results if a benchmarking is performed
  num_images_to_save: -1
  save_perturbation: false
  gpu: true
  output_path: "./benchmark_out" # Path to the output folder
  output_format: "report" # report or test

datasets:
  - name: "imagenet" # Name of the dataset
    num_classes: 1000
    subset: -1
    batch: 32
    type_dataset: 1
    num_workers: 2
    source_path: "../../data/imagenet-1k/val_images/"
    transform_config:
      size: 256
      crop: 224
      transform_id: "imagenet_like_crop"
      mean: [0.5074, 0.5308, 0.5306]
      std: [0.2639, 0.2518, 0.2521]

models:
  - name: 'resnetv2_50d_evos.ah_in1k'
    type: 'timm'
    pretrained: True
    num_classes: 1000
    task: "classification"
  - name: 'convnext_base.clip_laion2b_augreg_ft_in12k_in1k'
    type: 'timm'
    pretrained: True
    num_classes: 1000
    task: "classification"
  - name: 'vit_base_patch16_clip_224.openai_ft_in1k'
    type: 'timm'
    pretrained: True
    num_classes: 1000
    task: "classification"

attacks:
  - name: 'fgsm'
    id: 'another configuration'
    max_iters: 300
  - name: 'fgsm'
    id: 'a special configuration'
    max_iters: 20
  - name: 'banditprior'
    max_iters: 50
  - name: 'dag'
    max_iters: 50
```

As it was said before, some of the tested models in `timm` library are listed below 
* "m3bilenetv4\_conv\_blur\_medium.e500\_r224\_in1"
* "mobilenetv4\_conv\_medium.e500\_r224\_in1"
* "mobilenetv4\_hybrid\_medium.e500\_r224\_in1k"
* "resnet101.a1\_in1k"
* "resnet101.a1h\_in1k"
* "resnet101.a2\_in1k"
* "resnet101.gluon\_in1k"
* "resnet101.tv\_in1k"
* "resnet101c.gluon\_in1k"
* "resnet101d.gluon\_in1k"
* "resnet152.a1\_in1k"
* "resnet152.a1h\_in1k"
* "resnet152.a2\_in1k"
* "resnet152.gluon\_in1k"
* "resnet152.tv\_in1k"
* "resnet152c.gluon\_in1k"
* "resnet152d.gluon\_in1k"
* "resnet152s.gluon\_in1k"
* "resnetv2\_18.ra4\_e3600\_r224\_in1k"
* "resnetv2\_18d.ra4\_e3600\_r224\_in1k"
* "resnetv2\_34.ra4\_e3600\_r224\_in1k"
* "resnetv2\_34d.ra4\_e3600\_r224\_in1k"
* "resnetv2\_50.a1h\_in1k"
* "resnetv2\_50d\_evos.ah\_in1k"
* "resnetv2\_50d\_gn.ah\_in1k"
* "resnetv2\_50x1\_bit.goog\_distilled\_in1k"
* "resnetv2\_101.a1h\_in1k"

### Output folder
THe output folder is structured in the following way:
```
* output folder/
        * atks_info.json 			                # It contains the information of all the attacks.
        * name dataset/
            * model1/
                - examples/
                    * atk1/
                        - id1_ypred_yadvpred.png
                        - id1_pert.png
                        - ...
                    * atk2/
                        - id1_ypred_yadvpred.png
                        - id1_pert.png
                        - ...
                - data.json
            * model2/
                - attacks/
                    * atk1/
                        - adv1.png
                        - perturbation1.png
                        - ...
                    * atk2/
                        - adv1.png
                        - perturbation1.png
                        - ...
                - data.json
            * ...
            * examples/
                * id1_y1.png
                * id2_y2.png
                * ...
            * name_classes.json
```

