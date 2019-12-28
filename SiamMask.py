import cv2
import os
import numpy as np
import sys
from imutils.video import FPS
from PIL import Image
from skimage.transform import resize
import json

from trackers.tracker import SiamMask_Tracker, cfg
from utils.bbox_helper import cxy_wh_2_rect, xyxy_to_xywh
from mask_spline import mask2rect


def create_model(path):
    return SiamMask_Tracker(cfg, path)


def bb_on_im(im, point, size, mask):
    point = [int(l) for l in point]
    size = [int(l) for l in size]

    if len(mask):
        im[:, :, 2] = mask * 255 + (1 - mask) * im[:, :, 2]
    # prediction
    cv2.rectangle(im, (round(point[0] - size[0]/2), round(point[1] - size[1]/2)), (round(point[0] + size[0]/2), round(point[1] + size[1]/2)), (0, 255, 255), 3)

    return im


def track_object(model, state, mask, vid_path, framenum):
    vs = cv2.VideoCapture(vid_path)
    vs.set(1, framenum)
    ret, im = vs.read()
    '''import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.show
    plt.imshow(cv2.cvtColor(bb_on_im(im.copy(), *mask2rect(mask), mask), cv2.COLOR_BGR2RGB))
    plt.show()'''
    if type(state) == str:
        model = create_model(state)
        state = model.setup(im, *mask2rect(mask))

    state['mask'] = mask
    if not ret:
        return {'FINISHED'}, state, model
    _, im = vs.read()
    state = model.track(im, state)
    new_mask = state['mask'] > state['p'].seg_thr
    '''plt.imshow(cv2.cvtColor(bb_on_im(im, *mask2rect(new_mask), new_mask), cv2.COLOR_BGR2RGB))
    plt.title(state['score'])
    plt.show()'''
    return new_mask, state, model
