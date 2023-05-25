## action space
habitat-lab/habitat/config/default_structured_configs.py:1141  注册velocity_control
注：action space的box dtype是float32, 必须传入float32，否则判定action不包含在action space里

https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md  
下载需要的scene dataset和task dataset  

habitat-lab/habitat/tasks/nav/nav.py:1228  改scale，action[0]  

habitat-lab/habitat/utils/gym_adapter.py:300 改current dir  

[//]: # (habitat-lab/habitat/core/env.py:268 reset也拿info)



habitat-lab/habitat/core/env.py:114  
habitat-lab/habitat/tasks/nav/nav.py:1314  
habitat-lab/habitat/core/embodied_task.py:237  
habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py:258    创建环境

./data/default.physics_config.json 改dynamics  
habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py:283 加light  

{"episode_id": "0", "scene_id": "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb", "start_position": [3.9655439853668213, 0.17669875919818878, 1.5473179817199707], "start_rotation": [0, 0.9999815275096604, 0, 0.006078210217345751], "info": {"geodesic_distance": 2.269282102584839, "difficulty": "easy"}, "goals": [{"position": [3.8423774242401123, 0.17669875919818878, -0.7186191082000732], "radius": null}], "shortest_paths": null, "start_room": null}, 