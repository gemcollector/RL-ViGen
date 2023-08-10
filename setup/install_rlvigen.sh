pip install mujoco==2.3.0
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
pip install numpy==1.19.2
pip install protobuf==3.20.3
cd ../../../ && cd third_party/robosuite && pip install -r requirements.txt
cd ../../ & pip install -e ./envs/robosuiteVGB/
pip install h5py
pip install gym==0.23.1
pip install dotmap==1.3.30
pip install pygame==2.1.2
