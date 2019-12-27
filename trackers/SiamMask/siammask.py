# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Revised for THOR by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# relative imports
from .utils.log_helper import init_log, add_file_handler
from .utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from .utils.anchors import Anchors, generate_anchor
from .utils.tracker_config import TrackerConfig
from .utils.tracking_utils import get_subwindow_tracking

def SiamMask_init(im, target_pos, target_sz, model, hp=None):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    p = TrackerConfig()
    p.update(hp, model.anchors)
    p.renew()

    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = len(p.ratios) * len(p.scales)
    p.anchor = generate_anchor(model.anchors, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    use_cuda = torch.cuda.is_available()
    state['device'] = torch.device("cuda" if use_cuda else "cpu")
    state['p'] = p
    state['model'] = model
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['score'] = 1.0
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state

def SiamMask_track(state, im, temp_mem):
    p = state['p']
    avg_chans = state['avg_chans']
    window = state['window']
    old_pos = state['target_pos']
    old_sz = state['target_sz']
    dev = state['device']

    # get search area
    wc_x = old_sz[1] + p.context_amount * sum(old_sz)
    hc_x = old_sz[0] + p.context_amount * sum(old_sz)
    s_z = np.sqrt(wc_x * hc_x)

    scale_x = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_z + 2 * pad
    crop_box = [old_pos[0] - round(s_x) / 2, old_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, old_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # track
    target_pos, target_sz, score, best_id = temp_mem.batch_evaluate(x_crop.to(dev), old_pos,
                                                                old_sz, window,
                                                                scale_x, p)

    # mask refinement
    best_pscore_id_mask = np.unravel_index(best_id, (5, p.score_size, p.score_size))
    delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]
    mask = state['model'].track_refine((delta_y, delta_x)).to(dev).sigmoid().squeeze().view(
        p.out_size, p.out_size).cpu().data.numpy()

    def crop_back(image, bbox, out_sz, padding=-1):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    s = crop_box[2] / p.instance_size
    sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
               crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
               s * p.exemplar_size, s * p.exemplar_size]
    s = p.out_size / sub_box[2]
    back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
    mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

    target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]  # use max area polygon
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
        rbox_in_img = prbox
    else:  # empty mask
        location = cxy_wh_2_rect(target_pos, target_sz)
        rbox_in_img = np.array([[location[0], location[1]],
                                [location[0] + location[2], location[1]],
                                [location[0] + location[2], location[1] + location[3]],
                                [location[0], location[1] + location[3]]])

    state['mask'] = mask_in_img
    state['polygon'] = rbox_in_img

    # clip in min and max of the bb
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['crop'] = x_crop

    return state
