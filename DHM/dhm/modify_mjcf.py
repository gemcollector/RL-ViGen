from xml.dom.minidom import parse
import xmltodict
from dhm.utils import omegaconf_to_dict
from hydra import compose, initialize
import os
import numpy as np
# def modify_table_texture(src_mjcf_path, dest_mjcf_path, difficulty):
#     current_dir = os.path.dirname(__file__)
#     with open(current_dir + '/' + src_mjcf_path, 'r', encoding='utf-8') as f:
#         xml_str = f.read()
#     xml_dict = xmltodict.parse(xml_str)
#
#     xml_dict['mujocoinclude']['default']['geom']['@rgba'] = str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' 1'
#
#     xml_str = xmltodict.unparse(xml_dict)
#     with open(current_dir + '/' + dest_mjcf_path, 'w', encoding='utf-8') as f:
#         f.write(xml_str)

# def modify_objects_color(src_mjcf_path, dest_mjcf_path, color):
#     current_dir = os.path.dirname(__file__)
#     with open(current_dir + '/' + src_mjcf_path, 'r', encoding='utf-8') as f:
#         xml_str = f.read()
#     xml_dict = xmltodict.parse(xml_str)
#
#     if color is not None:
#         xml_dict['mujocoinclude']['default']['geom']['@rgba'] = str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' 1'
#
#     xml_str = xmltodict.unparse(xml_dict)
#     with open(current_dir + '/' + dest_mjcf_path, 'w', encoding='utf-8') as f:
#         f.write(xml_str)


def modify_assets(src_mjcf_path, dest_mjcf_path, color, table_texture, texture_idx):
    current_dir = os.path.dirname(__file__)
    with open(current_dir + '/' + src_mjcf_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    xml_dict = xmltodict.parse(xml_str)

    if color is not None:
        xml_dict['mujocoinclude']['default']['geom']['@rgba'] = str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' 1'
    if table_texture != 'original':
        for i in range(len(xml_dict['mujocoinclude']['asset']['material'])):
            if xml_dict['mujocoinclude']['asset']['material'][i]['@name'] == 'tablecube':
                xml_dict['mujocoinclude']['asset']['material'][i]['@texture'] = 'table_' + table_texture + str(texture_idx)
                break

    xml_str = xmltodict.unparse(xml_dict)
    with open(current_dir + '/' + dest_mjcf_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

def modify_mjcf_via_dict(src_mjcf_path, dest_mjcf_path, cfg_dict_task, random_state, cfg_dict_setting):
    current_dir = os.path.dirname(__file__)
    with open(current_dir + '/' + src_mjcf_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    xml_dict = xmltodict.parse(xml_str)
    xml_dict['mujoco']['include']['@file'] = f'DAPG_assets_test_{str(os.getpid())}.xml'
    # background
    if cfg_dict_task['background']['type'] != 'original' and cfg_dict_task['mode'] == 'test':
        del xml_dict['mujoco']['worldbody']['geom']
    else:
        del xml_dict['mujoco']['asset']['texture'][0]

    # hammer or pen or door(handle)
    if cfg_dict_task['mode'] == 'train':
        cfg_dict_task['object'] = 1
    if cfg_dict_task['taskdef']['task'] == 'hammer':
        count = 0
        for i in range(len(xml_dict['mujoco']['worldbody']['body'])):
            if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'Object':
                count += 1
                if cfg_dict_task['object'] == 1 :
                    if count == 2:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break
                elif cfg_dict_task['object'] == 2:
                    if count == 1:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break
    elif cfg_dict_task['taskdef']['task'] == 'door':
        for i in range(len(xml_dict['mujoco']['worldbody']['body'])):
            if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'frame':
                if cfg_dict_task['object'] == 1:
                    del xml_dict['mujoco']['worldbody']['body'][i]['body']['body']['geom'][2]
                    break
                elif cfg_dict_task['object'] == 2:
                    del xml_dict['mujoco']['worldbody']['body'][i]['body']['body']['geom'][1]
                    break
    elif cfg_dict_task['taskdef']['task'] == 'pen':
        count = 0
        for i in range(len(xml_dict['mujoco']['worldbody']['body'])):
            if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'Object':
                count += 1
                if cfg_dict_task['object'] == 1:
                    if count == 2:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break
                elif cfg_dict_task['object'] == 2:
                    if count == 1:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break
        count = 0
        for i in range(len(xml_dict['mujoco']['worldbody']['body'])):
            if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'target':
                count += 1
                if cfg_dict_task['object'] == 1:
                    if count == 2:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break
                elif cfg_dict_task['object'] == 2:
                    if count == 1:
                        del xml_dict['mujoco']['worldbody']['body'][i]
                        break

    # light
    # if cfg_dict_task['light_position'] == 'easy':
    #     xml_dict['mujoco']['worldbody']['light']['@pos'] = '-1.5 -2.0 4.0'
    # elif cfg_dict_task['light_position'] == 'hard':
    #     xml_dict['mujoco']['worldbody']['light']['@pos'] = '-3 -3 4.0'
    # if cfg_dict_task['light_color'] == 'easy':
    #     xml_dict['mujoco']['worldbody']['light']['@diffuse'] = '.7 .7 .5'
    # elif cfg_dict_task['light_color'] == 'hard':
    #     xml_dict['mujoco']['worldbody']['light']['@diffuse'] = '.7 .7 .3'
    # if cfg_dict_task['light']['intensity'] != 'original':
    if cfg_dict_task['mode'] == 'test':
        intensity = random_state.uniform(
            cfg_dict_setting['light']['intensity'][cfg_dict_task['light']['intensity']][0],
            cfg_dict_setting['light']['intensity'][cfg_dict_task['light']['intensity']][1])
        # if cfg_dict_task['light']['color'] != 'original':
        r = (cfg_dict_setting['origin_light_diffuse'][0] + random_state.uniform(
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][0],
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][1])) * intensity
        g = (cfg_dict_setting['origin_light_diffuse'][1] + random_state.uniform(
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][0],
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][1])) * intensity
        b = (cfg_dict_setting['origin_light_diffuse'][2] + random_state.uniform(
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][0],
            cfg_dict_setting['light']['color'][cfg_dict_task['light']['color']][1])) * intensity
        xml_dict['mujoco']['worldbody']['light']['@diffuse'] = f'{r} {g} {b}'

    # object pos and mass
    # if cfg_dict_task['mode'] == 'test':
        for i in range(len(xml_dict['mujoco']['worldbody']['body'])):
            if cfg_dict_task['taskdef']['task'] == 'hammer':
                if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'Object':
                    if cfg_dict_task['physical_properties'] == 'easy':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '0.3'
                    elif cfg_dict_task['physical_properties'] == 'hard':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '0.35'
            elif cfg_dict_task['taskdef']['task'] == 'door':
                if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'frame':
                    if cfg_dict_task['physical_properties'] == 'easy':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '8.85398'
                    elif cfg_dict_task['physical_properties'] == 'hard':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '10.85398'
            elif cfg_dict_task['taskdef']['task'] == 'pen':
                if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'Object':
                    if cfg_dict_task['physical_properties'] == 'easy':
                        xml_dict['mujoco']['worldbody']['body'][i]['geom'][0]['@density'] = '1400'
                    elif cfg_dict_task['physical_properties'] == 'hard':
                        xml_dict['mujoco']['worldbody']['body'][i]['geom'][0]['@density'] = '1200'
            elif cfg_dict_task['taskdef']['task'] == 'relocate':
                if xml_dict['mujoco']['worldbody']['body'][i]['@name'] == 'Object':
                    if cfg_dict_task['physical_properties'] == 'easy':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '0.279594'
                    elif cfg_dict_task['physical_properties'] == 'hard':
                        xml_dict['mujoco']['worldbody']['body'][i]['inertial']['@mass'] = '0.479594'

    xml_str = xmltodict.unparse(xml_dict)
    with open(current_dir + '/' + dest_mjcf_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)


if __name__ == '__main__':
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=[f"task={'hammer'}"])
        cfg_dict = omegaconf_to_dict(cfg)
    modify_mjcf_via_dict('../mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_hammer_template.xml',
                         '../mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_hammer_test.xml',
                         cfg_dict['task'])