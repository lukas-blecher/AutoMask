# --------------------------------------------------------
# THOR
# Licensed under The MIT License
# Written by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import os
from math import ceil, floor
from collections import deque
import abc

import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore

from .utils import to_numpy, print_color

TEMPLATE_SIZE = 127 # for visualization

class TemplateModule():
    def __init__(self, K, verbose, viz):
        self.verbose = verbose
        self.viz = viz
        self.is_full = False

        self._K = K
        self._templates_stack = None
        self._base_sim = 0
        self._gram_matrix = None

        if viz:
            # canvas to visualize the templates
            self._canvas_break = 5 # each row shows 5 templates max
            rows, cols = ceil(K/self._canvas_break), min(K, self._canvas_break)
            self.canvas = np.zeros((TEMPLATE_SIZE*rows, TEMPLATE_SIZE*cols, 3), dtype=np.uint8)

    def __len__(self):
        return self._K

    def pairwise_similarities(self, T_n, to_cpu=True):
        """
        calculate similarity of given template to all templates in memory
        """
        assert isinstance(T_n, torch.Tensor)
        sims = F.conv2d(T_n, self._templates_stack)
        if to_cpu:
            return np.squeeze(to_numpy(sims.data))
        else:
            return sims

    def _calculate_gram_matrix(self):
        dists = [self.pairwise_similarities(T, False) for T in self.templates['compare']]
        return [np.squeeze(to_numpy(d.data)) for d in dists]

    def _append_temp(self, temp):
        """
        append the given template to current memory
        """
        for k in temp.keys():
            self.templates[k].append(temp[k])

    def _set_temp(self, temp, idx):
        """
        switch out the template at idx
        """
        for k in temp.keys():
            self.templates[k][idx] = temp[k]
        self._templates_stack[idx, :, :, :] = temp['compare']

    def _update_canvas(self, temp, idx):
        """
        insert the template at given idx (transformed to row & col of canvas)
        """
        s_z = TEMPLATE_SIZE
        row, col = floor(idx/self._canvas_break), idx%self._canvas_break
        self.canvas[row*s_z:(row+1)*s_z,
                    s_z*col:s_z*(col + 1), :] = temp['im']

    @abc.abstractmethod
    def update(self, temp):
        """
        check if template should be taken into memory
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fill(self, temp):
        """
        fill all slots in the memory with the given template
        """
        raise NotImplementedError("Must be implemented in subclass.")

class ST_Module(TemplateModule):
    def __init__(self, K, template_keys, calc_div, verbose, viz):
        assert isinstance(K, int)
        if not K: return None
        super(ST_Module, self).__init__(K=K, verbose=verbose, viz=viz)
        self.templates = {key: deque(maxlen=K) for key in template_keys}
        self.calc_div = calc_div

    def _rebuild_canvas(self):
        """
        rebuild the full canvas with the current templates
        """
        self.canvas = np.concatenate(list(self.templates['im']), axis=1).astype(np.uint8)

    @staticmethod
    def normed_div_measure(t):
        """ calculate the normed diversity measure of t, the lower the more diverse """
        assert t.shape[0]==t.shape[1]
        dim = t.shape[0] - 1
        triu_no = int(dim/2*(dim + 1))
        return np.sum(np.triu(t, 1)) / (t[0,0] * triu_no)

    def _update_gram_matrix(self, temp):
        # calculate the current distance
        curr_sims = self.pairwise_similarities(temp['compare'])
        curr_sims = np.expand_dims(curr_sims, axis=1)

        # update gram matrix
        all_dists_new = np.block([[self._gram_matrix, curr_sims], [0, curr_sims.T]])

        # delete the row & col with idx 0 - the oldest template
        self._gram_matrix = all_dists_new[1:,1:]


    def fill(self, temp):
        for _ in range(self._K):
            self._append_temp(temp)

    def update(self, temp):
        """
        append to the current memory and rebuild canvas
        return div_scale (diversity of the current memory)
        """
        self._append_temp(temp)
        if self.viz: self._rebuild_canvas()

        # calculate diversity measure for the dynamic lower bound
        if self.calc_div:
            self._templates_stack =  torch.cat(list(self.templates['compare']))

            # calulate base dist_mat
            if not self.is_full:
                self._gram_matrix = np.squeeze(self._calculate_gram_matrix())
                self.is_full = True

            # update distance matrix and calculate the div scale
            self._update_gram_matrix(temp)

            return self.normed_div_measure(t=self._gram_matrix)
        else:
            return 1.0

class LT_Module(TemplateModule):
    def __init__(self, K, template_keys, lb, lb_type, verbose, viz):
        assert isinstance(K, int)
        super(LT_Module, self).__init__(K=K, verbose=verbose, viz=viz)

        self._K = K
        self._lb = lb
        self._lb_type = lb_type
        self._filled_idx = 0
        self.templates = {key: [] for key in template_keys}
        # self.save_det(np.array([np.nan]))

    def _throw_away_or_keep(self, curr_sims, self_sim, div_scale):
        """
        determine if we keep the template or not
        if the template is rejected: return -1 (not better) or -2 (rejected by lower bound)
        if we keep the template: return idx where to switch
        """
        base_sim = self._base_sim
        curr_sims = np.expand_dims(curr_sims, axis=1)

        # normalize the gram_matrix, otherwise determinants are huge
        gram_matrix_norm = self._gram_matrix/base_sim
        curr_sims_norm = curr_sims/base_sim

        # check if distance to base template is below lower bound
        if self._lb_type=='static':
            reject = (curr_sims[0] <  self._lb*base_sim)

        elif self._lb_type=='dynamic':
            lb = self._lb - (1 - div_scale)
            lb = np.clip(lb, 0.0, 1.0)
            reject = (curr_sims[0] <  lb*base_sim)

        elif self._lb_type=='ensemble':
            reject = not all(curr_sims_norm > self._lb)

        else:
            raise TypeError(f"lower boundary type {self._lb_type} not known.")

        if reject: return -2

        # fill the memory with adjacent frames if they are not
        # populated with something different than the base frame yet
        if self._filled_idx < (self._K-1):
            self._filled_idx += 1
            throwaway_idx = self._filled_idx

        # determine if and in which spot the template increases the current gram determinant
        else:
            curr_det = np.linalg.det(gram_matrix_norm)

            # start at 1 so we never throwaway the base template
            dets = np.zeros((self._K - 1))
            for i in range(self._K - 1):
                mat = np.copy(gram_matrix_norm)
                mat[i + 1, :] = curr_sims_norm.T
                mat[:, i + 1] = curr_sims_norm.T
                mat[i + 1, i + 1] = self_sim/base_sim
                dets[i] = np.linalg.det(mat)

            # check if any of the new combinations is better than the prev. gram_matrix
            max_idx = np.argmax(dets)
            if curr_det > dets[max_idx]:
                throwaway_idx = -1
            else:
                throwaway_idx = max_idx + 1

        assert throwaway_idx != 0
        return throwaway_idx if throwaway_idx != self._K else -1

    @staticmethod
    def save_det(d, p):
        if os.path.exists(p):
            old_det = np.load(p)
        else:
            old_det = np.array([])
        np.save(p, np.concatenate([old_det, d.reshape(-1)]))

    def _update_gram_matrix(self, curr_sims, self_sim, idx):
        """
        update the current gram matrix
        """
        curr_sims = np.expand_dims(curr_sims, axis=1)
        # add the self similarity at throwaway_idx spot
        curr_sims[idx] = self_sim

        self._gram_matrix[idx, :] = curr_sims.T
        self._gram_matrix[:, idx] = curr_sims.T

        # gram_matrix_norm = self._gram_matrix/self._base_sim
        # curr_det = np.linalg.det(gram_matrix_norm)
        # self.save_det(curr_det, 'determinants_dyn.npy')

    def fill(self, temp):
        for i in range(self._K):
            self._append_temp(temp)
            if self.viz: self._update_canvas(temp=temp, idx=i)

    def update(self, temp, div_scale):
        """
        decide if the templates is taken into the lt module
        """
        if not self.is_full:
            self._templates_stack =  torch.cat(self.templates['compare'])
            self._gram_matrix = np.squeeze(self._calculate_gram_matrix())
            self._base_sim = self._gram_matrix[0, 0]
            self.is_full = True

        # calculate the "throwaway_idx", the spot that the new template will take
        curr_sims = self.pairwise_similarities(temp['compare'])
        self_sim = F.conv2d(temp['compare'], temp['compare']).squeeze().item()
        throwaway_idx = self._throw_away_or_keep(curr_sims=curr_sims, self_sim=self_sim,
                                                 div_scale=div_scale)

        # if the idx is -2 or -1, the template is rejected, otherwise we update
        if throwaway_idx == -2:
            if self.verbose:
                print_color(f"DROPPED: too far from base template", Fore.BLUE)
        elif throwaway_idx == -1:
            if self.verbose:
                print_color(f"DROPPED: new template is not better", Fore.GREEN)
        else:
            if self.verbose:
                print_color(f"UPDATING: Switching Template "+
                            f"{throwaway_idx}", Fore.RED)
            self._set_temp(temp=temp, idx=throwaway_idx)
            self._update_gram_matrix(curr_sims=curr_sims, self_sim=self_sim, idx=throwaway_idx)
            if self.viz:
                self._update_canvas(temp=temp, idx=throwaway_idx)

class Dummy_Module():
    def __init__(self, template_keys):
        self.templates = {key: [] for key in template_keys}

    def __len__(self):
        return 0

    def fill(self, temp):
        return False

    def update(self, temp):
        return 1.0
