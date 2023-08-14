# Installation

Clone the RL-ViGen repo:
```
git clone https://github.com/gemcollector/RL-ViGen.git
cd RL-ViGen/
```

Create a conda environment:
```
conda create -n rl-vigen python=3.8
```
Run the installation script:
```
bash setup/install_rlvigen.sh
```

In addition, the following resources are required:
- CARLA: We apply CARLA 0.9.10 which is a stable version in RL-ViGen. The CARLA 0.9.10 should be downloaded first, and place it to the `./third_party` folder:
    ```
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.tar.gz
    ```
- Robosuite: we have employed the `1.4.0` version of [Robosuite](https://github.com/ARISE-Initiative/robosuite/), concurrently utilizing [mujoco](https://github.com/deepmind/mujoco) version `2.3.0` as the underlying simulator engine. We have incorporated all the relevant components associated with Robosuite in the first creating conda step.

 - DM-Control:  Our DM-Control also contains [mujoco_menagerie
](https://github.com/deepmind/mujoco_menagerie) as the basic component. We have incorporated all the relevant components associated with DM-Control in the first creating conda step.

To use the Habitat, you need to set up a separate conda environment:
```
conda create -n vigen-habitat python=3.8 cmake=3.14.0 -y 
```
```
bash setup/install_vigen-habitat.sh
```


1. We are using Gibson scene datasets for our experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

2. Next we need the episode dataset for the experiments. You can get the training and validation dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the ./data folder under the path : `data/datasets/pointnav/gibson/v1/`.


## Installation all in one

We provide a script to install all the dependencies and resources required for RL-ViGen. You can run the following command to install:

```
bash setup/install_all.sh
```
If you are employing this method of installation, please modify all the  `train` and `eval` scripts, ensuring that the parameter `task` is set to `task@_global_`. Here is an example:

```
CUDA_VISIBLE_DEVICES=0  python train.py \
						env=${env} \
						task@_global_=${task_name}
```


## Installation with Docker
```
docker pull cititude/rl-vigen:1.0
```