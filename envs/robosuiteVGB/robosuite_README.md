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
- `robots`:  specify your robot arm type.
- `camera_names`: change the camera in another view. 
- `changing_weather_speed`: 
- `vehicle`: the controlled vehicle type.



More infomation and details can be found at [Robosuite](https://robosuite.ai/docs/) official website.




