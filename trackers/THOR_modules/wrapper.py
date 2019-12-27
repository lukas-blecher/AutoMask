# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import abc
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.signal import tukey

from .utils import *
from .modules import ST_Module, LT_Module, Dummy_Module

MEDIATE_SIZE = 255

class THOR_Wrapper():
    def __init__(self, cfg, net):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._cfg = SimpleNamespace(**cfg)

        self._mem_len_total = self._cfg.K_st + self._cfg.K_lt
        assert self._cfg.K_lt > 0

        self.do_full_init = True
        self._net = net
        self._curr_type = 'lt'
        self.score_viz = None
        self.template_keys = ['im', 'raw', 'kernel', 'compare']

    def setup(self, im, pos, sz):
        """
        initialize the short-term and long-term module
        """
        self.avg_chans = np.mean(im, axis=(0, 1))
        self._frame_no = 0

        # make the template
        crop = self._get_crop(im, pos, sz)
        temp = self._make_template(crop)

        # initialize the short term module
        if self._cfg.K_st:
            self.st_module = ST_Module(K=self._cfg.K_st, template_keys=self.template_keys,
                                       calc_div=(self._cfg.lb_type=='dynamic'),
                                       verbose=self._cfg.verbose, viz=self._cfg.viz)
        else:
            self.st_module = Dummy_Module(self.template_keys)
        self.st_module.fill(temp)

        # initialize the long term module
        if self.do_full_init or self._cfg.vanilla:
            self.lt_module = LT_Module(K=self._cfg.K_lt, template_keys=self.template_keys,
                                       lb=self._cfg.lb, lb_type=self._cfg.lb_type,
                                       verbose=self._cfg.verbose, viz=self._cfg.viz)
            self.lt_module.fill(temp)
            self.do_full_init = False
        else:
            # reinitialize long term only at the beginning of the episode
            self.lt_module.update(temp, div_scale=0)

    def update(self, im, curr_crop, pos, sz):
        """
        update the short-term and long-term module and
        update the shown templates and activations (score_viz)
        """
        self._frame_no += 1

        # only update according to dilation steps
        if not self._frame_no%self._cfg.dilation:
            crop = self._get_crop(im, pos, sz)
            temp = self.crop_to_mem(crop)

            # reset st if it drifted
            if self._cfg.K_st and self._curr_type=='lt':
                self.st_module.fill(temp)

        if self._cfg.viz:
            self._show_modulate(torch_to_img(curr_crop), self.score_viz)
            self._show_templates('st')
            self._show_templates('lt')

    def crop_to_mem(self, crop):
        """
        make the template and insert into modules
        """
        temp = self._make_template(crop)

        # temp to st and lt module
        div_scale = self.st_module.update(temp)
        if self._cfg.K_lt > 1:
            self.lt_module.update(temp, div_scale=div_scale)

        return temp

    def _get_best_temp(self, pos, sz, score):
        """
        determine the best template and return the prediction and the
        score of the best long-term template
        """
        # get the best score in st and lt memory
        score_st, score_lt = np.split(score, [self._cfg.K_st])
        best_st = [] if not len(score_st) else np.argmax(score_st)
        best_lt = np.argmax(score_lt) + self._cfg.K_st

        # calculate iou and switch to lt if iou too low
        iou = self.get_IoU(pos.T[best_st], sz.T[best_st], pos.T[best_lt], sz.T[best_lt])
        self._curr_type = 'lt' if iou < self._cfg.iou_tresh else 'st'

        return (best_lt if self._curr_type=='lt' else best_st), score[best_lt]

    def _show_templates(self, mode='lt'):
        if mode=='st' and not self._cfg.K_st: return
        mem = self.st_module if mode=='st' else self.lt_module
        y_plot = 50 if mode=='st' else 300

        temp_canvas = mem.canvas.copy()
        cv2.imshow(f"Templates {mode}", temp_canvas)
        cv2.moveWindow(f"Templates {mode}", 1200, y_plot)

    @staticmethod
    def get_IoU(pos_1, sz_1, pos_2, sz_2):
        if not len(pos_1): return 0.0 # st memory is empty
        if not len(pos_2): return 1.0 # lt memory is empy
        return IOU_numpy(xywh_to_xyxy(np.append(pos_1, sz_1)), \
                         xywh_to_xyxy(np.append(pos_2, sz_2)))

    @staticmethod
    def modulate(score, mem_len, out_sz):
        """
        modulate the prediction of each template with a mean activation map of all templates
        """
        score_per_temp = int(np.prod(score.shape) / (mem_len * np.prod(out_sz)))
        score_im = score.reshape(mem_len, score_per_temp, *out_sz)
        score_mean = np.mean(score_im, axis=1)

        #modulation according to score:
        weights = np.max(score_mean, axis=(1, 2))
        weights = weights.reshape(len(weights), 1, 1)
        score_mean *= weights
        # modulate the mean with the weights
        score_mean_all = np.mean(score_mean, axis=0).reshape(1, *out_sz)
        score_mean_norm = score_mean_all/np.max(score_mean_all)

        # modulate: multiply with the mean
        mean_tiled = np.tile(score_mean_norm.reshape(1, -1), score_per_temp)
        score = score*mean_tiled
        return score, score_mean_norm

    @staticmethod
    def _show_modulate(im, score_viz):
        """
        show the current activations on top of the current crop
        """
        if score_viz is None: return # modulation is not active

        im = cv2.resize(im, (MEDIATE_SIZE, MEDIATE_SIZE)).astype(np.uint8)
        canvas = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

        # calculate the color map
        score_im_base = cv2.resize(score_viz[0], im.shape[:2])
        score_im_base = (255*score_im_base).astype(np.uint8)
        im_color = cv2.applyColorMap(score_im_base, cv2.COLORMAP_JET)

        # show the image
        overlayed_im = cv2.addWeighted(im, 0.8, im_color, 0.7, 0)
        canvas[:, :im.shape[1], :] = overlayed_im
        cv2.imshow('modulated', canvas)
        cv2.moveWindow('modulated', 1200, 800)

    @abc.abstractmethod
    def custom_forward(self, x):
        """
        implements the forward pass through the network of the tracker
        with an added batch dimension [tracker specific]
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def _get_crop(self, im, pos, sz):
        """
        get the crop from the search window [tracker specific]
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def _make_template(self, crop):
        """
        given a crop, make a template [tracker specific]
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def batch_evaluate(self, crop):
        """
        take evalue method from original tracker and add batch processing for all
        templates in memory and add modulating [tracker specific]
        """
        raise NotImplementedError("Must be implemented in subclass.")

class THOR_SiamFC(THOR_Wrapper):
    def __init__(self, cfg, net):
        super(THOR_SiamFC, self).__init__(cfg, net)
        self.template_sz = 127
        self.kernel_sz = 6
        self.max_response = 0

    def _get_crop(self, im, pos, sz):
        context_sz = self._cfg.context_temp * np.sum(sz)
        crop = get_subwindow_tracking_SiamFC(im=im, pos=pos, model_sz=self.template_sz,
                                             context_sz=context_sz, avg_chans=self.avg_chans,
                                             target_sz=sz)
        return crop.unsqueeze(0)

    def _make_template(self, crop):
        temp = {}
        temp['raw'] = crop.to(self.device)
        temp['im'] = torch_to_img(crop)
        temp['kernel'] = self._net.feature(temp['raw'])

        # add the tukey window to the temp for comparison
        alpha = self._cfg.tukey_alpha
        win = np.outer(tukey(self.kernel_sz, alpha), tukey(self.kernel_sz, alpha))
        temp['compare'] = temp['kernel'] * torch.Tensor(win).to(self.device)
        return temp

    def custom_forward(self, x):
        x_f = self._net.feature(x) # 3 x 256 x 22 x 22
        kernel_cat = torch.cat(list(self.st_module.templates['kernel']) + \
                           list(self.lt_module.templates['kernel'])) # mem_len x 256 x 22 x 22

        # convolve
        out = F.conv2d(x_f, kernel_cat).permute(1, 0, 2, 3) # mem_len x 3 x 17 x 17

        # adjust the scale of the responses
        return out * 0.001

    def batch_evaluate(self, crop, old_pos, old_sz, p):
        # get responses
        responses = self.custom_forward(crop)
        responses = responses.data.cpu().numpy()
        batch_sz, scales = responses.shape[:2]


        # upscale
        upscale = lambda im: cv2.resize(im, (p.upscale_sz, p.upscale_sz),
                                        interpolation=cv2.INTER_CUBIC)
        responses = np.array([[upscale(responses[t, s]) for s in range(scales)] for t in range(batch_sz)])

        responses[:, :p.scale_num // 2] *= p.penalty_k
        responses[:, p.scale_num // 2 + 1:] *= p.penalty_k

        # get peak scale for every template
        scale_ids = np.argmax(np.amax(responses, axis=(2, 3)), axis=1)

        # apply penalty
        responses = responses[np.arange(batch_sz), scale_ids, :, :]
        responses -= np.min(responses, axis=(1,2)).reshape(-1,1,1)
        responses /= np.sum(responses, axis=(1,2)).reshape(-1,1,1)+ 1e-16
        responses = (1 - p.window_influence) * responses + \
                        p.window_influence * p.hann_window

        # mediating
        if self._cfg.modulate:
            old_shape = responses.shape
            responses = responses.reshape(batch_sz, -1)
            responses, self.score_viz = self.modulate(responses, self._mem_len_total, old_shape[-2:])
            responses = responses.reshape(*old_shape)

        # get the peak idcs
        get_peak_idx = lambda x: np.unravel_index(x.argmax(), x.shape)
        locs = [get_peak_idx(t) for t in responses]

        # locate target center
        disp_in_response = np.array(locs) - p.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            p.total_stride / p.response_up
        disp_in_image = disp_in_instance * p.x_sz * \
                p.scale_factors[scale_ids].reshape(-1, 1) / p.instance_sz
        target_pos = old_pos + disp_in_image

        # update target size
        scale = (1 - p.lr) * 1.0 + \
            p.lr * p.scale_factors[scale_ids].reshape(-1, 1)
        target_sz = old_sz*scale

        # normalize the scores to the score of the initial frame
        best_scores = np.max(responses, axis=(1,2))
        if not self.max_response:
            self.max_response = best_scores[0]
            best_scores = np.ones_like(best_scores)
        else:
            best_scores /= self.max_response
            best_scores = np.clip(best_scores, 0, 1)

        # determine the currently best template
        best_temp, lt_score = self._get_best_temp(target_pos.T, target_sz.T, best_scores)
        return target_pos[best_temp], target_sz[best_temp], lt_score, scale[best_temp]

class THOR_SiamRPN(THOR_Wrapper):
    def __init__(self, cfg, net):
        super(THOR_SiamRPN, self).__init__(cfg, net)
        self.template_sz = 127
        self.kernel_sz = 6
        self.template_keys += ['reg', 'cls', 'reg_anc', 'cls_anc']
        self.curr_temp = None

    def _get_crop(self, im, pos, sz):
        wc_z = sz[0] + self._cfg.context_temp * sum(sz)
        hc_z = sz[1] + self._cfg.context_temp * sum(sz)
        context_size =  round(np.sqrt(wc_z * hc_z))

        crop = get_subwindow_tracking_SiamRPN(im=im, pos=pos, model_sz=self.template_sz,
                                             original_sz=context_size,
                                             avg_chans=self.avg_chans)
        return crop.unsqueeze(0)

    def _make_template(self, crop):
        temp = {}
        temp['raw'] = crop.to(self.device)
        temp['im'] = torch_to_img(crop)

        temp['kernel'] = self._net.featureExtract(temp['raw'])
        temp['reg'] = self._net.conv_r1(temp['kernel'])
        temp['cls'] = self._net.conv_cls1(temp['kernel'])
        t_s = temp['reg'].data.size()[-1]

        temp['reg_anc'] = temp['reg'].view(self._net.anchor*4, self._net.feature_out, t_s, t_s)
        temp['cls_anc'] = temp['cls'].view(self._net.anchor*2, self._net.feature_out, t_s, t_s)

        # add the tukey window to the temp for comparison
        alpha = self._cfg.tukey_alpha
        win = np.outer(tukey(self.kernel_sz, alpha), tukey(self.kernel_sz, alpha))
        temp['compare'] = temp['kernel'] * torch.Tensor(win).to(self.device)
        return temp

    def custom_forward(self, x):
        x_f = self._net.featureExtract(x)

        def reg_branch(x, reg_cat, l):
            out = F.conv2d(x, reg_cat)
            out = out.view(l, out.shape[1]//l, out.shape[2], out.shape[3])
            return out

        def cls_branch(x, cls_cat, l):
            out = F.conv2d(x, cls_cat)
            return out.view(l, out.shape[1]//l, out.shape[2], out.shape[3])

        # regression
        x_reg = self._net.conv_r2(x_f)
        reg_cat = torch.cat(list(self.st_module.templates['reg_anc']) + \
                            list(self.lt_module.templates['reg_anc']))
        reg_res = reg_branch(x_reg, reg_cat, self._mem_len_total)
        reg_res = self._net.regress_adjust(reg_res)

        # classification
        x_cls = self._net.conv_cls2(x_f)
        cls_cat = torch.cat(list(self.st_module.templates['cls_anc']) + \
                            list(self.lt_module.templates['cls_anc']))
        cls_res = cls_branch(x_cls, cls_cat, self._mem_len_total)

        return reg_res, cls_res, x_f

    def batch_evaluate(self, crop, pos, size, window, scale_z, p):
        """
        adapted from SiamRPNs tracker_evaluate
        """
        delta, score, x_f = self.custom_forward(crop)
        out_sz = score.shape[-2:]
        batch_sz = self._mem_len_total

        delta = delta.contiguous().view(batch_sz, 4, -1).data.cpu().numpy()
        score = F.softmax(score.contiguous().view(batch_sz, 2, -1), dim=1).data[:, 1, :].cpu().numpy()

        # delta regression
        anc = np.tile(p.anchor, (batch_sz, 1, 1))
        delta[:, 0, :] = delta[:, 0, :] * anc[:, :, 2] + anc[:, :, 0]
        delta[:, 1, :] = delta[:, 1, :] * anc[:, :, 3] + anc[:, :, 1]
        delta[:, 2, :] = np.exp(delta[:, 2, :]) * anc[:, :, 2]
        delta[:, 3, :] = np.exp(delta[:, 3, :]) *anc[:, :, 3]

        # penalizing
        def change(r):
            return np.maximum(r, 1./r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # scale penalty
        s_c = change(sz(delta[:, 2, :], delta[:, 3, :]) / (sz_wh(size)))
        # ratio penalty
        r_c = change((size[0] / size[1]) / (delta[:, 2, :] / delta[:, 3, :]))

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # mediating
        if self._cfg.modulate:
            pscore, self.score_viz = self.modulate(pscore, self._mem_len_total, out_sz)

        # target regression
        best_pscore_id = np.argmax(pscore, axis=1)
        # arange is needed for correct indexing
        target = delta[np.arange(batch_sz), :, best_pscore_id] / scale_z
        target_sz = size / scale_z
        lr = penalty[np.arange(batch_sz), best_pscore_id] *\
                score[np.arange(batch_sz), best_pscore_id] * p.lr

        res_x = target[:, 0] + pos[0]
        res_y = target[:, 1] + pos[1]
        res_w = target_sz[0] * (1 - lr) + target[:, 2] * lr
        res_h = target_sz[1] * (1 - lr) + target[:, 3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        best_scores = pscore[np.arange(batch_sz), best_pscore_id]

        # determine the currently best template
        best_temp, lt_score = self._get_best_temp(target_pos, target_sz, best_scores)
        return np.squeeze(target_pos[:, best_temp]), np.squeeze(target_sz[:, best_temp]), lt_score

class THOR_SiamMask(THOR_Wrapper):
    def __init__(self, cfg, net):
        super(THOR_SiamMask, self).__init__(cfg, net)
        self.template_sz = 127
        self.kernel_sz = 7

    def _get_crop(self, im, pos, sz):
        wc_z = sz[0] + self._cfg.context_temp * sum(sz)
        hc_z = sz[1] + self._cfg.context_temp * sum(sz)
        context_size =  round(np.sqrt(wc_z * hc_z))

        crop = get_subwindow_tracking_SiamRPN(im=im, pos=pos, model_sz=self.template_sz,
                                             original_sz=context_size,
                                             avg_chans=self.avg_chans)
        return crop.unsqueeze(0)

    def _make_template(self, crop):
        temp = {}
        temp['raw'] = crop.to(self.device)
        temp['im'] = torch_to_img(crop)
        temp['kernel'] = self._net.template(temp['raw'])

        # add the tukey window to the temp for comparison
        alpha = self._cfg.tukey_alpha
        win = np.outer(tukey(self.kernel_sz, alpha), tukey(self.kernel_sz, alpha))
        temp['compare'] = temp['kernel'] * torch.Tensor(win).to(self.device)
        return temp

    def custom_forward(self, x):
        self._net.zf = torch.cat(list(self.st_module.templates['kernel']) + \
                                 list(self.lt_module.templates['kernel']))
        pred_cls, pred_loc, _ = self._net.track_mask(x)
        return pred_loc, pred_cls

    def batch_evaluate(self, crop, pos, size, window, scale_x, p):
        """
        adapted from SiamRPNs tracker_evaluate
        """
        delta, score = self.custom_forward(crop)

        out_sz = score.shape[-2:]
        batch_sz = self._mem_len_total

        delta = delta.contiguous().view(batch_sz, 4, -1).data.cpu().numpy()
        score = F.softmax(score.contiguous().view(batch_sz, 2, -1), dim=1).data[:, 1, :].cpu().numpy()

        # delta regression
        anc = np.tile(p.anchor, (batch_sz, 1, 1))
        delta[:, 0, :] = delta[:, 0, :] * anc[:, :, 2] + anc[:, :, 0]
        delta[:, 1, :] = delta[:, 1, :] * anc[:, :, 3] + anc[:, :, 1]
        delta[:, 2, :] = np.exp(delta[:, 2, :]) * anc[:, :, 2]
        delta[:, 3, :] = np.exp(delta[:, 3, :]) *anc[:, :, 3]

        # penalizing
        def change(r):
            return np.maximum(r, 1./r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # scale penalty
        target_sz_in_crop = size*scale_x
        s_c = change(sz(delta[:, 2, :], delta[:, 3, :]) / (sz_wh(target_sz_in_crop)))
        # ratio penalty
        r_c = change((size[0] / size[1]) / (delta[:, 2, :] / delta[:, 3, :]))

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # mediating
        if self._cfg.modulate:
            pscore, self.score_viz = self.modulate(pscore, self._mem_len_total, out_sz)

        # target regression
        best_pscore_id = np.argmax(pscore, axis=1)
        # arange is needed for correct indexing
        target = (delta[np.arange(batch_sz), :, best_pscore_id] / scale_x)
        lr = penalty[np.arange(batch_sz), best_pscore_id] *\
                score[np.arange(batch_sz), best_pscore_id] * p.lr
        target, lr = target.astype(np.float64), lr.astype(np.float64)

        res_x = target[:, 0] + pos[0]
        res_y = target[:, 1] + pos[1]
        res_w = size[0] * (1 - lr) + target[:, 2] * lr
        res_h = size[1] * (1 - lr) + target[:, 3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        best_scores = pscore[np.arange(batch_sz), best_pscore_id]

        # determine the currently best template
        best_temp, lt_score = self._get_best_temp(target_pos, target_sz, best_scores)
        self._net.best_temp = best_temp

        return np.squeeze(target_pos[:, best_temp]), np.squeeze(target_sz[:, best_temp]), \
                lt_score, best_pscore_id[best_temp]
