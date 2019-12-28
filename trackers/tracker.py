# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# Modified by Lukas Blecher
# --------------------------------------------------------

from os.path import dirname, abspath, join
import torch
from trackers.THOR_modules.wrapper import THOR_SiamMask

# SiamMask Imports
from trackers.SiamMask.net import SiamMaskCustom
from trackers.SiamMask.siammask import SiamMask_init, SiamMask_track
from trackers.SiamMask.utils.load_helper import load_pretrain

cfg = {'tracker': {'window_influence': 0.42,
                   'instance_size': 255,
                   'base_size': 8,
                   'out_size': 127,
                   'seg_thr': 0.3,
                   'penalty_k': 0.04,
                   'lr': 0.25},
       'anchors': {'stride': 8,
                   'ratios': [0.33, 0.5, 1, 2, 3],
                   'scales': [8],
                   'round_dight': 0},
       'THOR': {'K_st': 6,
                'K_lt': 3,
                'iou_tresh': 0.742568,
                'lb': 0.27996,
                'tukey_alpha': 0.697998,
                'lb_type': 'ensemble',
                'modulate': True,
                'dilation': 10,
                'context_temp': 0.5,
                'viz': False,
                'verbose': False,
                'vanilla': False}}


class Tracker():
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.mask = False
        self.temp_mem = None

    def init_func(self, im, pos, sz):
        raise NotImplementedError

    def track_func(self, state, im):
        raise NotImplementedError

    def setup(self, im, target_pos, target_sz):
        state = self.init_func(im, target_pos, target_sz)
        self.temp_mem.setup(im, target_pos, target_sz)
        return state

    def track(self, im, state):
        state = self.track_func(state, im)
        self.temp_mem.update(im, state['crop'], state['target_pos'], state['target_sz'])
        return state


class SiamMask_Tracker(Tracker):
    def __init__(self, cfg, proj_path=dirname(abspath(__file__))):
        super(SiamMask_Tracker, self).__init__()
        self.cfg = cfg
        self.mask = True

        # setting up the model
        model = SiamMaskCustom(anchors=cfg['anchors'])
        model = load_pretrain(model, join(proj_path, 'trackers/SiamMask/model.pth'))
        self.model = model.eval().to(self.device)

        # set up template memory
        self.temp_mem = THOR_SiamMask(cfg=cfg['THOR'], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamMask_init(im, pos, sz, self.model, self.cfg['tracker'])

    def track_func(self, state, im):
        return SiamMask_track(state, im, self.temp_mem)
