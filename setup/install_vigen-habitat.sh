conda install habitat-sim==0.2.3 withbullet headless -c conda-forge -c aihabitat -y
cd envs/habitatVGB/ && pip install -e habitat-lab
cd ../../ && pip install -e envs/habitatVGB/
pip install mujoco==2.3.0
pip install dm-control==1.0.8
pip install glfw==1.12.0
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tb-nightly -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install termcolor==1.1.0
pip install scipy==1.10.0
pip install ipdb
pip install wandb
pip install hydra-core==1.3.1
pip install hydra-submitit-launcher==1.1.5
pip install omegaconf==2.3.0
