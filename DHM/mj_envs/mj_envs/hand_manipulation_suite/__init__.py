from gym.envs.registration import register
# from mjrl.envs.mujoco_env import MujocoEnv


## adroit
# Swing the door open
register(
    id='door-adroit-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorAdroitEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_adroit_v0 import DoorAdroitEnvV0

# Hammer a nail into the board
register(
    id='hammer-adroit-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerAdroitEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_adroit_v0 import HammerAdroitEnvV0

# Reposition a pen in hand
register(
    id='pen-adroit-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenAdroitEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_adroit_v0 import PenAdroitEnvV0

# Relcoate an object to the target
register(
    id='relocate-adroit-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateAdroitEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_adroit_v0 import RelocateAdroitEnvV0

## mpl
# Swing the door open
register(
    id='door-mpl-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorMplEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_mpl_v0 import DoorMplEnvV0

# Hammer a nail into the board
register(
    id='hammer-mpl-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerMplEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_mpl_v0 import HammerMplEnvV0

# Reposition a pen in hand
register(
    id='pen-mpl-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenMplEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_mpl_v0 import PenMplEnvV0

# Relcoate an object to the target
register(
    id='relocate-mpl-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateMplEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_mpl_v0 import RelocateMplEnvV0


## barrett
# Swing the door open
register(
    id='door-barrett-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorBarrettEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_barrett_v0 import DoorBarrettEnvV0

# Hammer a nail into the board
register(
    id='hammer-barrett-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerBarrettEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_barrett_v0 import HammerBarrettEnvV0

# Reposition a pen in hand
register(
    id='pen-barrett-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenBarrettEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_barrett_v0 import PenBarrettEnvV0

# Relcoate an object to the target
register(
    id='relocate-barrett-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateBarrettEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_barrett_v0 import RelocateBarrettEnvV0


## shadowhand
# Swing the door open
# register(
#     id='door-shadowhand-v0',
#     entry_point='mj_envs.hand_manipulation_suite:DoorShadowhandEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.door_shadowhand_v0 import DoorShadowhandEnvV0

# # Hammer a nail into the board
# register(
#     id='hammer-shadowhand-v0',
#     entry_point='mj_envs.hand_manipulation_suite:HammerShadowhandEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.hammer_shadowhand_v0 import HammerShadowhandEnvV0

# # Reposition a pen in hand
# register(
#     id='pen-shadowhand-v0',
#     entry_point='mj_envs.hand_manipulation_suite:PenShadowhandEnvV0',
#     max_episode_steps=100,
# )
# from mj_envs.hand_manipulation_suite.pen_shadowhand_v0 import PenShadowhandEnvV0

# # Relcoate an object to the target
# register(
#     id='relocate-shadowhand-v0',
#     entry_point='mj_envs.hand_manipulation_suite:RelocateShadowhandEnvV0',
#     max_episode_steps=200,
# )
# from mj_envs.hand_manipulation_suite.relocate_shadowhand_v0 import RelocateShadowhandEnvV0



