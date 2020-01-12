import cv2
import os
import numpy as np
import sys
from PIL import Image

from trackers.tracker import SiamMask_Tracker, cfg
from mask_spline import mask2rect


def create_model(path):
    return SiamMask_Tracker(cfg, path)


def track_object(model, state, mask, vid_path, framenum):
    vs = cv2.VideoCapture(vid_path)
    vs.set(1, framenum-1)
    ret, im = vs.read()
    im = im[..., :3]
    if type(state) == str:
        model = create_model(state)
        state = model.setup(im, *mask2rect(mask))
    state['mask'] = mask
    if not ret:
        return None, state, model
    _, im = vs.read()
    im = im[..., :3]
    state = model.track(im, state)
    new_mask = state['mask'] > state['p'].seg_thr
    return new_mask, state, model
