# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import cv2
import torch
import numpy as np
from colorama import Fore, Style

# numpy - torch conversions

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def torch_to_img(img):
    img = to_numpy(torch.squeeze(img, 0))
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img

# tracker specific functions

def get_subwindow_tracking_SiamRPN(im, pos, model_sz, original_sz, avg_chans):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch)

def get_subwindow_tracking_SiamFC(im, pos, model_sz, context_sz, avg_chans, target_sz):
    # convert box to corners (0-indexed)
    z_sz = np.sqrt(np.prod(target_sz + context_sz))
    size = round(z_sz)

    corners = np.concatenate((
        np.round(pos.T - (size.T - 1) / 2) -1,
        np.round(pos.T - (size.T - 1) / 2) -1 + size))
    corners = np.round(corners).astype(int)
    # bring in correct order
    corners = np.array([corners[1], corners[0], corners[3], corners[2]])

    # pad imare if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - im.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        im = cv2.copyMakeBorder(
            im, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=avg_chans)

    # crop image patch
    corners = (corners + npad).astype(int)
    im_patch_original = im[corners[0]:corners[2], corners[1]:corners[3]]

    im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    return im_to_torch(im_patch)

def get_subwindow_tracking_SiamRPN_PP(im, pos, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                         int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                      int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    # im_patch = im_patch.astype(np.float32)
    # im = im_to_torch(im_patch)


    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    return im_patch

# Misc
LIMIT = 99999999
def xywh_to_xyxy(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    xMid = bboxes[0,...]
    yMid = bboxes[1,...]
    width = bboxes[2,...]
    height = bboxes[3,...]
    bboxesOut[0,...] = xMid - width / 2.0
    bboxesOut[1,...] = yMid - height / 2.0
    bboxesOut[2,...] = xMid + width / 2.0
    bboxesOut[3,...] = yMid + height / 2.0
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    return bboxesOut

def print_color(str_, color="Fore.RED"):
    print (color + str_ + f"{Style.RESET_ALL}")

def IOU_numpy(rect1, rect2):
    x1s = np.fmax(rect1[0], rect2[0])
    x2s = np.fmin(rect1[2], rect2[2])
    y1s = np.fmax(rect1[1], rect2[1])
    y2s = np.fmin(rect1[3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, .00001)
    return intersection * 1.0 / union
