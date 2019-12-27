# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h


LIMIT = 99999999
def xyxy_to_xywh(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    x1 = bboxes[0,...]
    y1 = bboxes[1,...]
    x2 = bboxes[2,...]
    y2 = bboxes[3,...]
    bboxesOut[0,...] = (x1 + x2) / 2.0
    bboxesOut[1,...] = (y1 + y2) / 2.0
    bboxesOut[2,...] = x2 - x1
    bboxesOut[3,...] = y2 - y1
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    return bboxesOut
