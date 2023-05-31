# Habitat Instruction

## Config file
The config file is located at `RL-ViGen/envs/habitatVGB/cfg/config.yaml`. You can change the config file for different setups.




## Testing script
```
from habitat import *     
mode = 'train'
seed = 1
appearance_id = 'original'
name = "HabitatImageNav-v0"

env = make_env(name, mode=mode, seed=seed, appearance_id=appearance_id)
obs = env.reset()
```
you can run `mytest.py` to test the environment as well. 

The scene and task datasets can be donwloaded from https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md.

- We are using Gibson scene datasets for our experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

- Next we need the episode dataset for the experiments. You can get the training and validation dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) and place it in the ./data folder under the path : `data/datasets/pointnav/gibson/v1/`.


## Parameter Description

### *Visual Appearances*
- `appearance_id:` the scenarios with different visual appearances. The training mode is 'original' while the evaluation mode is filled with an integer.

### *Lighting Changes*
- `light_position`: change the position of the light.
- `light_color`: change the color of the light.
- `light_intensity`: change the intensity of the light.

### *Camera Views*
- `position`: change the camera position.
- `orientation`: change the camera orientation.
- `hfov`: change the camera hfov.

### *Scene Structures*
- `scene_id`: different maps and scenes. 





More infomation and details can be found at [Habitat](https://www.habitat.co.uk/) official website.