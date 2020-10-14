import cv2
import os
import numpy as np
import sys
from PIL import Image

from trackers.tracker import SiamMask_Tracker, cfg
from mask_spline import mask2rect


class ImageReader:
    is_video = True

    def __init__(self, movie_details, framenum):
        print(movie_details)
        if movie_details['source'] == 'SEQUENCE':
            self.is_video = False
            dirname = os.path.dirname(movie_details['path'])
            imgs = os.listdir(dirname)
            ind = imgs.index(os.path.basename(movie_details['path']))
            self.imgs = [os.path.join(dirname, i) for i in imgs[ind:]]
            self.i = framenum-1
        else:
            self.vs = cv2.VideoCapture(movie_details['path'])
            self.vs.set(1, framenum-1)
        print(self.is_video)

    def read(self):
        if self.is_video:
            return self.vs.read()
        else:
            if self.i+1 > len(self.imgs):
                return False, cv2.imread(self.imgs[-1])
            else:
                self.i += 1
                return True, cv2.imread(self.imgs[self.i-1])


def create_model(path):
    return SiamMask_Tracker(cfg, path)


def track_object(model, state, mask, movie_details, framenum):
    vs = ImageReader(movie_details, framenum)
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
