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
### Systematic params
- `port`: The port of the CARLA engine.
- `traffic_port`: The port of the CARLA traffic manager.
- `rl_image_size`: The image size of the input image.

### Environment params
- `fov`:
- `cameras`:
- `map`:
- `weather`: 
- `changing_weather_speed`: 
- `vehicle`: the controlled vehicle type.








