## install  
安装robosuite  
pip install -r requirements.txt  
pip install -e .  

cd  
## 测试脚本test/env_test.py  
  
## use   
在cfg/config.yaml中配置env  

可参考 test/env_test.py  
from robosuitvgb import *  
env = make_env()   

或  
import robosuitvgb      
env = robosuitvgb.make_env()  

env.step()  
env.reset()  





