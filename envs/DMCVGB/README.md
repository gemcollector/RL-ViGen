# DMC Instruction
   


## Config file
The config file is located at `RL-ViGen/envs/DMCVGB/cfg/config.yaml`. You can change the config file for different setups.


## Testing script
```
from dmcvgb import *     
env = make_env(domain_name='unitree', task_name='walk', seed=1)
```
you can run `test/env_test.py` to test the environment as well. 



Distracting Control Suite uses the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset for video backgrounds, which can be downloaded by running

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```


## Parameter Description


### *Visual Appearances*
- `is_distracting_cs`: whether to apply distracting control suite.
- `intensity`： the intensity of the distracting control suite.
- `background`: change the overall visual appearances with different types and levels.
- `objects_color`: change the color of the objects or embodiments.

### *Lighting Changes*
- `light_position`: change the position of the light.
- `light_color`: change the color of the light.
- `moving_light`: whether to apply moving light with different moving speeds.

### *Camera Views*
- `cam_id`: cameras at different views.
- `cam_pos`: change the camera position.

### *Cross Embodiments*
The embodiment can be changed via running different task. For example, switching `domain_name` from `unitree` to `anymal` will change the embodiment.

More infomation and details can be found at [DM-Control](https://robosuite.ai/docs/) official website, [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control) and [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) repo.