pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
cd mujoco-py && pip install -e .
pip install patchelf==0.17.2.0 gym==0.21.0
pip install dm-control==0.0.425341097
pip install absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image \
    termcolor==1.1.0 imageio-ffmpeg==0.4.4 hydra-core==1.1.0 hydra-submitit-launcher==1.1.5 ipdb==0.13.9 \
    yapf==0.31.0 opencv-python==4.5.3.56 psutil tb-nightly

pip install joblib==1.2.0
cd .. 
cd DHM && pip install -e .
cd mjrl && pip install -e .
cd ../mj_envs
pip install -e .
pip install glfw==1.12.0
pip install xmltodict