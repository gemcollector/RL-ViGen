conda create -n rl-vigen-all python=3.8.16 cmake=3.14.0 -y
conda activate rl-vigen-all
cd RL-ViGen/ && pip install mujoco==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ./envs/DMCVGB
pip install -e ./envs/DMCVGB/dmc2gym
pip install pyparsing==2.4.7
pip install dm_control==1.0.8
pip uninstall dm_control -y
cd envs/DMCVGB/dm_control && python setup.py install
pip install xmltodict==0.13.0
pip install glfw==1.12.0
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install termcolor==1.1.0
pip install tb-nightly
pip install imageio==2.9.0
pip install imageio-ffmpeg==0.4.4
pip install hydra-core==1.1.0
pip install hydra-submitit-launcher==1.1.5
pip install pandas==1.3.0
pip install ipdb==0.13.9
pip install yapf==0.31.0
pip install sklearn==0.0
pip install matplotlib==3.4.2
pip install wandb==0.11.1
pip install moviepy==1.0.3
pip install opencv-python==4.5.3.56
pip install scipy==1.9.3
pip install protobuf==3.20.3
cd ../../../ && cd third_party/robosuite && pip install -r requirements.txt
cd ../../ && pip install -e ./envs/robosuiteVGB/
pip install h5py
pip install gym==0.23.1
pip install dotmap==1.3.30
pip install pygame==2.1.2
git checkout -b ViGen-adroit origin/ViGen-adroit
cd mujoco-py && pip install -e .
pip install absl-py==0.13.0 jupyterlab==3.0.14 scikit-image \
    imageio-ffmpeg==0.4.4 hydra-core==1.1.0 ipdb==0.13.9 \
    yapf==0.31.0 psutil
# https://mirrors.aliyun.com/pypi/simple
pip install joblib==1.2.0
cd ..
cd DHM && pip install -e .
cd mjrl && pip install -e .
cd ../mj_envs
pip install -e .
conda install habitat-sim==0.2.3 withbullet headless -c conda-forge -c aihabitat -y
git checkout RL-ViGen
cd envs/habitatVGB/ && pip install -e habitat-lab
cd ../../ && pip install -e envs/habitatVGB/
pip install hydra-core==1.3.1
pip install hydra-submitit-launcher==1.1.5
pip install omegaconf==2.3.0
pip install gym==0.23.0
cd third_party/robosuite && pip install -r requirements.txt
cd ../../ && pip install -e ./envs/robosuiteVGB/