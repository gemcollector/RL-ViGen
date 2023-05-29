# DMC Instruction
   
  
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


### Environment params
- `is_distracting_cs`: whether to apply distracting control suite.
- `intensity`： the intensity of the distracting control suite.
- `cam_id`: cameras at different views.
- ``





More infomation and details can be found at [DM-Control](https://robosuite.ai/docs/) official website, [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control) and [DMC-GB](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) repo.