# Adroit Instruction

To get started, you should download adroit demos, pretrained models, and example logs with the following link to create `vrl3data` folder: https://drive.google.com/drive/folders/14rH_QyigJLDWsacQsrSNV7b0PjXOGWwD?usp=sharing. In addtion, you should change the `local_data_dir` in each `cfg/config.yaml` to the path of `vrl3data`.

## Setup
- Create a conda env:
    ```
    conda create -n vigen-adroit python=3.8
    ```
- run the `install_adroit.sh`:
    ```
    bash install_adroit.sh
    ```

## Config file
The config file is located at `ViGen-Adroit/DHM/cfg/task`. You can change the config file for different setups.
In addtion, you should change the `cfg/aug_config.cfg` to your own path to load the extra augmented images for some algorithms, and modify the `hydra/run/dir` to log your training data.

## Training
```
cd src/
bash train.sh
```


## Evaluation
```
cd src/
model_dir=/path/to/model
bash adroit_eval.sh test_vrl3_color-easy  #{mode}_{agent_name}_{generalization type}
```
You should change the `model_dir` to your own path to load the trained model. 

## Parameter Description

### *Visual Appearances*
- `background`: change the overall visual appearances with different types and levels.

- `object_color`: change the color of the object.

- `table_texture`: change the texture of the table.


### *Lighting Changes*
- `moving_light`: whether to apply moving light.

-  `light`: light in Adroit can be adjusted with `position`, `intensity`, and `color` with various difficulty level.

### *Camera Views*
- `camera`: camera in Adroit can be adjusted with `position`, `orientation`, and `fov` with various difficulty level.

### *Cross Embodiments*
- `hand`: change the embodiment of the hand with various categroies.


More infomation and details can be found at [mjrl](https://github.com/aravindr93/mjrl) and [VRL3](https://github.com/microsoft/VRL3). 