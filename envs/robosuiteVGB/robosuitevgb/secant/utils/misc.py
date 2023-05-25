import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import cv2  # pip install opencv-python==4.2.0.*


def disable_gym_warnings(disable: bool = True):
    import gym, logging

    if disable:
        gym.logger.setLevel(logging.ERROR)
    else:
        gym.logger.setLevel(logging.WARNING)


def set_os_envs(envs: Optional[Dict[str, Any]] = None):
    """
    Special value __delete__ indicates that the ENV_VAR should be removed
    """
    if envs is None:
        envs = {}
    # check for special key __delete__
    for k, v in envs.items():
        if v == "__delete__":
            os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in envs.items() if v != "__delete__"})


def render_img(img, backend="cv2", waitkey=100):
    if backend == "matplotlib":
        plt.imshow(img, aspect="auto")
        plt.show()
    elif backend == "cv2":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("rendering", img)
        cv2.waitKey(waitkey)
    else:
        raise AssertionError("only matplotlib and cv2 are supported.")

        
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name='null', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
