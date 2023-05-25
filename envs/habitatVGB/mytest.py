import gym
import numpy as np
from matplotlib import pyplot as plt
import habitat
from habitat import *
# import habitat.utils.gym_definitions
# import habitat.utils.make_env
from habitat_sim.utils import viz_utils as vut

video_file_path = "data/example_interact.mp4"
video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
env = gym.make("HabitatCloseCab-v0")

# env = gym.make("HabitatImageNav-v0")
# env = gym.make("HabitatInstanceImageNav-v0")
# env = gym.make("HabitatObjectNav-v0")
# env = gym.make("HabitatPointNav-v0")

# env = habitat.utils.make_env.make_env("HabitatImageNav-v0")
# env = habitat.Env(config=habitat.get_config("benchmark/nav/imagenav/imagenav_test.yaml"))

done = False
count = 0
obs = env.reset()
obs_rgb = obs['robot_head_depth']
obs, reward, done, info = env.step(env.action_space.sample())
while (not done) and count < 1000:
    # aaa = env.action_space.contains(np.array([6.0, 0.2], dtype=np.float32))  # action space的box dtype是float32, 必须传入float32，否则判定action不包含在action space里
    obs, reward, done, info = env.step(env.action_space.sample())
    plt.figure(dpi=300)
    plt.imshow(env.render("rgb_array"))
    plt.show()
    video_writer.append_data(env.render(mode="rgb_array"))
    count += 1

video_writer.close()

env.close()


# import habitat
# from habitat.sims.habitat_simulator.actions import HabitatSimActions
# import cv2
#
#
# FORWARD_KEY="w"
# LEFT_KEY="a"
# RIGHT_KEY="d"
# FINISH="f"
#
#
# def transform_rgb_bgr(image):
#     return image[:, :, [2, 1, 0]]
#
#
# def example():
#     env = habitat.Env(
#         config=habitat.get_config("benchmark/nav/imagenav/imagenav_test.yaml")
#     # config = habitat.get_config(
#     #     "benchmark/nav/pointnav/pointnav_habitat_test.yaml")
#     )
#
#     print("Environment creation successful")
#     observations = env.reset()
#     # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
#     #     observations["pointgoal_with_gps_compass"][0],
#     #     observations["pointgoal_with_gps_compass"][1]))
#     cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
#
#     print("Agent stepping around inside environment.")
#
#     count_steps = 0
#     while not env.episode_over:
#         keystroke = cv2.waitKey(0)
#
#         if keystroke == ord(FORWARD_KEY):
#             action = HabitatSimActions.move_forward
#             print("action: FORWARD")
#         elif keystroke == ord(LEFT_KEY):
#             action = HabitatSimActions.turn_left
#             print("action: LEFT")
#         elif keystroke == ord(RIGHT_KEY):
#             action = HabitatSimActions.turn_right
#             print("action: RIGHT")
#         elif keystroke == ord(FINISH):
#             action = HabitatSimActions.stop
#             print("action: FINISH")
#         else:
#             print("INVALID KEY")
#             continue
#
#         observations = env.step(action)
#         count_steps += 1
#
#         # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
#         #     observations["pointgoal_with_gps_compass"][0],
#         #     observations["pointgoal_with_gps_compass"][1]))
#         cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
#
#     print("Episode finished after {} steps.".format(count_steps))
#
#     if (
#         action == HabitatSimActions.stop
#         and observations["pointgoal_with_gps_compass"][0] < 0.2
#     ):
#         print("you successfully navigated to destination point")
#     else:
#         print("your navigation was unsuccessful")
#
#
# if __name__ == "__main__":
#     example()
