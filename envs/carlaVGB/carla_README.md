# CARLA Instruction

## Config file
The config file is located at `RL-ViGen/cfgs/carlaenv10_config.yaml`(for training) and `RL-ViGen/cfgs/carlaenv10_eval_config.yaml`(for evaluation). You can change the config file for different setups.

## Testing script
```
from carlaenv.utils import make_env_10
action_repeat = 2
env = make_env_10(action_repeat)
```

## Parameter Description
### Systematic params
- `port`: The port of the CARLA engine.
- `traffic_port`: The port of the CARLA traffic manager.
- `rl_image_size`: The image size of the input image.

### Environment params


#### *Visual Appearances and Light Changes*
- `weather`: Change the driving weather.
- `changing_weather_speed`: the speed of weather change.


#### *Camera Views*
- `fov`: change the camera's fov. 
- `cameras`: cameras with different views. 


#### *Scene Structures*
- `map`: change the dirving map. 
- `scenario`: change the evaluation scneraios. 
- `num_other_cars`: change the number of other cars in the maps. 
- `num_other_cars_nearby`: change the number of other cars near our controlled cars. 
- `vehicle_spawn_point_id`: change the intialized positions of the cars. 

#### *Cross Embodiments*
- `vehicle`: the controlled vehicle type.


### Visualization
Since the CARLA engine is based on Unreal Engine, we recommend you to render the images with a monitor. You can set the `render_display` param to 1 to render the images.




More infomation and details can be found at [CARLA](http://carla.org/) official website.
