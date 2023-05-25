import gym.spaces


def map_gym_space(fn, space):
    """
    Recursively transform Dict and Tuple spaces

    Args:
        fn: apply to any space that is not Dict or Tuple
    """
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: map_gym_space(fn, v) for k, v in space.spaces.items()}
        )
    elif isinstance(space, gym.spaces.Tuple):
        return gym.spaces.Tuple([map_gym_space(fn, v) for v in space.spaces])
    else:
        return fn(space)


def wrap_dict_tuple_space(space):
    """
    Turn a regular dict or list/tuple object into gym.spaces.Dict and Tuple
    """
    if isinstance(space, dict):
        return gym.spaces.Dict({k: wrap_dict_tuple_space(s) for k, s in space.items()})
    elif isinstance(space, (list, tuple,)):
        return gym.spaces.Tuple([wrap_dict_tuple_space(s) for s in space])
    else:
        return space
