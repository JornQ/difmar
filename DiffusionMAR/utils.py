import argparse
import importlib

import torch
import torchvision.transforms as T
from PIL import Image
import os

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict

def create_gif(frames, location, name, duration):
    transform = T.ToPILImage()
    frames_pil = []
    for frame in frames:
        frames_pil.append(transform(frame.squeeze()))
    frame_one = frames_pil[0]
    frame_one.save(os.path.join(location, name), format="GIF", append_images=frames_pil,
               save_all=True, duration=duration, loop=0)