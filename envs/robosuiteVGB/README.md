# Robosuite Instruction

## Config file
The config file is located at `RL-ViGen/envs/robosuiteVGB/cfg/robo_config.yaml`. You can change the config file for different setups.



## Testing script
```
from robosuitevgb import *
env = make_env(task_name='Door', seed=1)
```
you can run `test/env_test.py` to test the environment as well. 

## Parameter Description


### Environment params
- `task_def`: specify the running task name.
-  `mode`: specify the running mode, `train` or `eval` with various difficult levels.




#### *Visual Appearances*
- `scene_id`: change the scene appearances.
- `video_background`: whether to apply dynamic video backgrounds.
- `except_robot`: whether to change the robot appearances.

#### *Lighting Changes*
- `moving_light`: whether to apply moving light.
- `light`: `light` in Robosuite contains a series of parameters, including `position`, `direction`, `diffuse`, etc. .


#### *Camera Views*
- `camera_names`: change the camera in another view. 
- `camera`: `camera` in Robosuite contains a series of parameters, including `position`, `rotation`, `fovy`, etc. .

#### *Cross Embodiments*
- `robots`:  specify your robot arm type.



## Task Description
In our paper, three tasks including single-arm and dual-arm settings are selected in RL-ViGen: **Door**, **Lift**, and **TwoArmPegInhole**. Additionally, we create multiple difficulty levels, incorporating various visual scenarios, dynamic backgrounds, aiming for more comprehensively evaluating the agent's generalization ability. 






More infomation and details can be found at our paper [Robosuite](https://robosuite.ai/docs/) official website.




