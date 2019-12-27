import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.load_helper import load_pretrain
from .utils.anchors import Anchors
from .resnet import resnet50

# basic model

class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.all_anchors = None

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc

# rpn

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    ## WRAPPER: changed, otherwise it does not work with batches
    # x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    x = x.expand(batch, *x.shape[1:])
    x = x.contiguous().view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))

    return out

class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out

# mask

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

# additional modules

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l, r = 4, -4
            x = x[:, :, l:r, l:r]
        return x

class ResDown(Features):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3

class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)

class Refine(nn.Module):
    def __init__(self):
        """
        Mask refinement module
        Please refer SiamMask (Appendix A)
        https://arxiv.org/abs/1812.05050
        """
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos=None):
        pos = [int(i) for i in pos]
        p0 = torch.nn.functional.pad(f[0], [16,16,16,16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        p1 = torch.nn.functional.pad(f[1], [8,8,8,8])[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]
        p2 = torch.nn.functional.pad(f[2], [4,4,4,4])[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]

        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out

# final siammask model

class SiamMaskCustom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(SiamMaskCustom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.best_temp = 0

    def refine(self, f, pos=None):
        return self.refine_model(f, pos)

    def template(self, template):
        self.zf = self.features(template)
        return self.zf

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        ### WRAPPER
        self.corr_feature = self.corr_feature[self.best_temp].unsqueeze(0)
        ###
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos)
        return pred_mask
