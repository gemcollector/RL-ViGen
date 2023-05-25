import os
import random
import copy
import numpy as np
import gym.spaces
from gym.spaces import Box, Dict
from ...wrappers import wrap_dict_tuple_space


def get_xml_file(xml_path):
    if os.path.isfile(xml_path):
        return xml_path
    elif os.path.isdir(xml_path):
        xmls = []
        for filename in os.listdir(xml_path):
            if filename.endswith("xml"):
                xmls.append(filename)
        if len(xmls) == 0:
            raise FileNotFoundError(
                f"Directory {xml_path} does not contain any XML file."
            )

        fname = random.choice(xmls)
        path = os.path.join(xml_path, fname)
        return path
    else:
        raise FileExistsError(f"Path {xml_path} does not exist.")


def _gen_obs_space(obs):
    return gym.spaces.Box(high=np.inf, low=-np.inf, shape=obs.shape, dtype=obs.dtype)


def get_obs_shape_from_dict(obs_dict):
    import tree

    return wrap_dict_tuple_space(tree.map_structure(_gen_obs_space, obs_dict))
