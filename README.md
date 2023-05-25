

# RL-ViGen: A Reinforcement Learning Benchmark for visual generalization
[Website](https:///gemcollector.github.io/gvrlb/) | [Paper]

We have released RL-ViGen -- a **R**einforcement **L**earning Benchmark for **Vi**sual **Gen**eralization. It specifically specifically designs to tackle the visual generalization problem.
Our benchmark comprises five major task categories, namely, manipulation, navigation, autonomous driving, locomotion, and dexterous manipulation. Concurrently, it encompasses five classes of generalization types: visual appearances, camera views, lighting changes, scene structures, and cross-embodiments. Furthermore, RL-ViGen integrates seven of the most currently favored visual RL and generalization algorithms, solidifying its broad applicability in the field.


<p align="center">
  <br />
  <a href="./MODEL_CARD.md"><img alt="Model Card" src="https://img.shields.io/badge/benchmark-RLViGen-green.svg" /></a>
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <a href="Python 3.8"><img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg" /></a>
  <a href="https://github.com/psf/black"><img alt="Code style: white" src="https://img.shields.io/badge/code%20style-white-white.svg" /></a>
</p>




<p align="center">
  <img width="85.5%" src="./img/all_bench.png">

[comment]: <> (  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">)

[comment]: <> (  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">)
 </p>

[comment]: <> (## Method)

[comment]: <> (DrQ-v2 is a model-free off-policy algorithm for image-based continuous control. DrQ-v2 builds on [DrQ]&#40;https://github.com/denisyarats/drq&#41;, an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements including:)

[comment]: <> (- Switch the base RL learner from SAC to DDPG.)

[comment]: <> (- Incorporate n-step returns to estimate TD error.)

[comment]: <> (- Introduce a decaying schedule for exploration noise.)

[comment]: <> (- Make implementation 3.5 times faster.)

[comment]: <> (- Find better hyper-parameters.)

[comment]: <> (<p align="center">)

[comment]: <> (  <img src="https://i.imgur.com/SemY10G.png" width="100%"/>)

[comment]: <> (</p>)

[comment]: <> (These changes allow us to significantly improve sample efficiency and wall-clock training time on a set of challenging tasks from the [DeepMind Control Suite]&#40;https://github.com/deepmind/dm_control&#41; compared to prior methods. Furthermore, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL.)

[comment]: <> (<p align="center">)

[comment]: <> (  <img width="100%" src="https://imgur.com/mrS4fFA.png">)

[comment]: <> (  <img width="100%" src="https://imgur.com/pPd1ks6.png">)

[comment]: <> ( </p>)


## Intallation
To install our benchmark, please follow the instructions in [INSTALLATION.md](INSTALLATION.md).
<!-- 
### Create a conda environment
This basic conda env can serve for DM-Control, CARLA and Robosuite.
```
bash install_conda.sh
``` -->

<!-- ### Installation of CARLA
We apply CARLA 0.9.10 which is a stable version in our benchmark.

- you should download the CARLA 0.9.10 first: 
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.tar.gz
```


### Installation of Robosuite
we have employed the `1.4.0` version of Robosuite, concurrently utilizing mujoco version `2.3.0` as the underlying simulator engine.
we have incorporated all the relevant components associated with Robosuite in the first creating conda step.

### Installation of Habitat
Since Habitat applies a different `hydra` version, we need to create a new conda environment for Habitat.
```
bash install_habitat.sh
```
1. We are using Gibson scene datasets for our experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

2. Next we need the episode dataset for the experiments. You can get the training and validation dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the ./data folder under the path : `data/datasets/pointnav/gibson/v1/`.




### Installation of Adroit
Our Adroit implementation is originated from [VRL3](https://github.com/microsoft/VRL3).

In case the mujoco rendering bug, we recommand you to create a new adroit conda env for your experiment as follow:
```
bash install_adroit.sh
``` -->


## Training

<!-- ### CARLA
- For CARLA, you should start the CARLA engine first:
```
SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -fps 20 --carla-port=2022
```
The `SDL_HINT_CUDA_DEVICE` is the running GPU id, and we can set the `carla-port` to run multiple CARLA instances.

- Third, you should run the code as the following way:
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 bash carlatrain.sh
``` -->

### Habitat, Robosuite, Locomotion
```
bash train.sh
```
### CARLA
```
bash carlatrain.sh
```

## Evaluation
For `Robosuite` , `Habitat`, and `CARLA`, we can run the evaluation code as follow:
```
bash eval.sh 
```
You should change the `env`, `task_name`, `test_agent` for different evaluation in the `eval.sh`.

For `DM-Control`, we can run the evaluation code as follow:
```
bash locoeval.sh
```

For more details, please refer to the `README.md` files for each environment in the `env/` directory.


## Acknowledgements
Our training code is based on [DrQv2](https://github.com/facebookresearch/drqv2). 
And we also thank the codebase of [VRL3](https://github.com/microsoft/VRL3), [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark), and [SECANT](https://github.com/DrJimFan/SECANT), and [kyoran](https://github.com/kyoran/CarlaEnv-Benchmark).

## License
The majority of DrQ-v2, DMCGB, VRL3 is licensed under the MIT license. Habitat Lab, dmc2gym, mujoco-py are also licensed under the MIT license. however portions of the project are available under separate license terms: DeepMind,  mj_envs, and mjrl is licensed under the Apache 2.0 license.Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/). CARLA specific assets are distributed under CC-BY License. The ad-rss-lib library compiled and linked by the [RSS Integration build variant](Docs/adv_rss.md) introduces [LGPL-2.1-only License](https://opensource.org/licenses/LGPL-2.1). Unreal Engine 4 follows its [own license terms](https://www.unrealengine.com/en-US/faq).
