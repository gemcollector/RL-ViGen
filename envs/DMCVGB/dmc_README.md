## install
conda env create -f env.yml  
conda activate dmcvgb    
或  
conda activate yourexistingenv  
conda env update --file env.yml  

cd dm_control  
python setup.py install  
cd ..  
cd dmc2gym  
python setup.py install  
cd ..  
pip install -e .
   
  
## 测试脚本test/env_test.py  
  
## use   
在cfg中配置env  

可参考 test/env_test.py  
from dmcvgb import *     
env = make_env()  

或
import dmcvgb    
env = dmcvgb.make_env()  

env.step()  
env.reset()  




Distracting Control Suite uses the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset for video backgrounds, which can be downloaded by running

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```


