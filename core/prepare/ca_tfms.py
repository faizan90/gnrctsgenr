'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from math import factorial
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from fcopulas import (
    asymms_exp,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,
    get_etpy_min_nd,
    get_etpy_max_nd,
    )

from ...misc import (
    get_local_entropy_ts_cy,
    )


class GTGPrepareTfms:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self):

        return

    def _get_probs(self, data, make_like_ref_flag=False):

        probs_all = np.empty_like(data, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            probs = rankdata(data[:, i]) / (data.shape[0] + 1.0)

            if make_like_ref_flag:
                assert self._rr.probs_srtd is not None

                probs = self._rr.probs_srtd[
                    np.argsort(np.argsort(probs)), i]

            probs_all[:, i] = probs

        return probs_all

    def _get_asymm_1_max(self, scorr):

        return get_asymm_1_max(scorr)

    def _get_asymm_2_max(self, scorr):

        return get_asymm_2_max(scorr)

    def _get_etpy_min(self, n_bins):

        return get_etpy_min(n_bins)

    def _get_etpy_max(self, n_bins):

        return get_etpy_max(n_bins)

    def _get_etpy_min_nd(self, n_bins):

        return get_etpy_min_nd(n_bins, self._data_ref_shape[1])

    def _get_etpy_max_nd(self, n_bins):

        return get_etpy_max_nd(n_bins, self._data_ref_shape[1])

    def _get_cumm_ft_corr(self, ref_ft, sim_ft):

        '''
        There are no predefined norming values here.
        '''

        ref_mag = np.abs(ref_ft)
        ref_phs = np.angle(ref_ft)

        sim_mag = np.abs(sim_ft)
        sim_phs = np.angle(sim_ft)

        numr = (
            ref_mag[1:,:] *
            sim_mag[1:,:] *
            np.cos(ref_phs[1:,:] - sim_phs[1:,:]))

        demr = (
            ((ref_mag[1:,:] ** 2).sum(axis=0) ** 0.5) *
            ((sim_mag[1:,:] ** 2).sum(axis=0) ** 0.5))

        return np.cumsum(numr, axis=0) / demr

    def _get_auto_cumm_corrs_ft(self, mag_spec, vtype, data_type):

        '''
        Auto cumm. corr.

        mag_spec can be of the data or probs.
        '''

        mag_spec = mag_spec[1:,:]

        cumm_pwrs = (mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (data_type == 'data'):
            norm_vals = self._rr.data_ft_norm_vals

        elif (vtype == 'sim') and (data_type == 'probs'):
            norm_vals = self._rr.probs_ft_norm_vals

        elif (vtype == 'ref') and (data_type == 'data'):
            norm_vals = cumm_pwrs[-1,:].copy().reshape(1, -1)
            self._rr.data_ft_norm_vals = norm_vals

        elif (vtype == 'ref') and (data_type == 'probs'):
            norm_vals = cumm_pwrs[-1,:].copy().reshape(1, -1)
            self._rr.probs_ft_norm_vals = norm_vals

        else:
            raise NotImplementedError

        cumm_pwrs /= norm_vals

        return cumm_pwrs

    def _get_gnrc_ft(self, data, vtype):

        assert data.ndim == 1

        ft = np.fft.rfft(data)
        mag_spec = np.abs(ft)

        mag_spec_cumsum = np.concatenate(
            ([ft.real[0]], mag_spec[1:].cumsum()))

        if vtype == 'sim':
            norm_val = None
            sclrs = None
            frst_term = None

        elif vtype == 'ref':
            frst_term = mag_spec_cumsum[0]

            norm_val = float(mag_spec_cumsum[-1])

            mag_spec_cumsum[1:] /= norm_val

            mag_spec_cumsum[:1] = 1

            # sclrs lets the first few long amplitudes into account much
            # better. These describe the direction i.e. Asymmetries.
#             sclrs = 1.0  / np.arange(1.0, mag_spec_cumsum.size + 1.0)
            sclrs = mag_spec / norm_val
            sclrs[0] = 1.0  # sclrs[1:].sum()

        else:
            raise NotImplementedError

        return (mag_spec_cumsum, norm_val, sclrs, frst_term)

    def _get_ms_cross_pair_ft(self, mags_spec, phss_spec, vtype, data_type):

        '''
        Pairwise cross cummulative correlation spectrum with phases.
        '''

        assert mags_spec.ndim == 2

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        mags_spec = mags_spec[1:,:]
        phss_spec = phss_spec[1:,:]

        max_comb_size = 2  # self._data_ref_n_labels

        if vtype == 'ref':
            norm_vals = []
            pwr_spec_sum_sqrt = (mags_spec ** 2).sum(axis=0) ** 0.5

        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            n_combs = int(
                factorial(self._data_ref_n_labels) /
                (factorial(comb_size) *
                 factorial(self._data_ref_n_labels - comb_size)))

            pair_cumm_corrs = np.empty(
                ((self._data_ref_shape[0] // 2), n_combs))

            for i, comb in enumerate(combs):
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError('Configured for pairs only!')

                numr = (
                    mags_spec[:, col_idxs[0]] *
                    mags_spec[:, col_idxs[1]] *
                    np.cos(phss_spec[:, col_idxs[0]] -
                           phss_spec[:, col_idxs[1]])
                    )

                pair_cumm_corrs[:, i] = numr

                if vtype == 'ref':
                    demr = (
                        pwr_spec_sum_sqrt[col_idxs[0]] *
                        pwr_spec_sum_sqrt[col_idxs[1]])

                    norm_vals.append(demr)

            break  # Should only happen once due to the pair comb case.

        if vtype == 'ref':
            norm_vals = np.array(norm_vals).reshape(1, -1)

            n_combs = int(
                factorial(self._data_ref_n_labels) /
                (factorial(max_comb_size) *
                 factorial(self._data_ref_n_labels - max_comb_size)))

            assert norm_vals.size == n_combs, (norm_vals.size, n_combs)

            if data_type == 'data':
                self._rr.data_ms_pair_ft_norm_vals = norm_vals

            elif data_type == 'probs':
                self._rr.probs_ms_pair_ft_norm_vals = norm_vals

            else:
                raise NotImplementedError

        elif vtype == 'sim':
            if data_type == 'data':
                norm_vals = self._rr.data_ms_pair_ft_norm_vals

            elif data_type == 'probs':
                norm_vals = self._rr.probs_ms_pair_ft_norm_vals

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        pair_cumm_corrs = np.cumsum(pair_cumm_corrs, axis=0)

        pair_cumm_corrs /= norm_vals

        return pair_cumm_corrs

    def _get_ms_cross_cumm_corrs_ft(self, mags_spec, vtype, data_type):

        '''
        Multivariate cross cummulative maximum correlation.
        '''

        assert mags_spec.ndim == 2

        cumm_corrs = mags_spec.prod(axis=1).cumsum()

        if (vtype == 'sim') and (data_type == 'data'):
            norm_val = self._rr.data_ms_ft_norm_val

        elif (vtype == 'sim') and (data_type == 'probs'):
            norm_val = self._rr.probs_ms_ft_norm_val

        elif vtype == 'ref':
            norm_val = (
                (cumm_corrs ** mags_spec.shape[1]).sum(axis=0).prod()
                ) ** (1.0 / mags_spec.shape[1])

            if data_type == 'data':
                self._rr.data_ms_ft_norm_val = norm_val

            elif data_type == 'probs':
                self._rr.probs_ms_ft_norm_val = norm_val

        else:
            raise NotImplementedError

        cumm_corrs /= norm_val

        return cumm_corrs

    def _get_gnrc_mult_ft(self, data, vtype, tfm_type):

        '''
        IDEA: How about finding the best matching pairs for the reference
        case? And then having individual series. Because, some series might
        not have correlations and result in zero combined variance.

        The ultimate test for this would be to do a rainfall runoff sim
        with the annealed series.
        '''

        data = data.copy(order='f')

        assert tfm_type in ('asymm1', 'asymm2', 'corr', 'etpy')

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        ft_inputs = []

        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            n_combs = int(
                factorial(self._data_ref_n_labels) /
                (factorial(comb_size) *
                 factorial(self._data_ref_n_labels - comb_size)))

            input_specs = np.empty(
                ((self._data_ref_shape[0] // 2) + 1, n_combs))

            for i, comb in enumerate(combs):
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError('Configured for pairs only!')

                if tfm_type == 'asymm1':
                    # Asymmetry 1.
                    # data is probs.
                    ft_input = (
                        (data[:, col_idxs[0]] +
                         data[:, col_idxs[1]] -
                         1.0) ** asymms_exp)

                elif tfm_type == 'asymm2':
                    # Asymmetry 2.
                    # data is probs.
                    ft_input = (
                        (data[:, col_idxs[0]] -
                         data[:, col_idxs[1]]) ** asymms_exp)

                elif tfm_type == 'corr':
                    ft_input = (
                        (data[:, col_idxs[0]] *
                         data[:, col_idxs[1]]))

                elif tfm_type == 'etpy':
                    ft_input = get_local_entropy_ts_cy(
                        data[:, col_idxs[0]],
                        data[:, col_idxs[1]],
                        self._sett_obj_ecop_dens_bins)

                else:
                    raise NotImplementedError

                ft_inputs.append(ft_input)

                ft = np.fft.rfft(ft_input)
                mag_spec = np.abs(ft)

                # For the multipair case, mag_spec is needed.
                input_spec = mag_spec  # ** 2

                if ft.real[0] < 0:
                    input_spec[0] *= -1

                input_specs[:, i] = input_spec

            break  # Should only happen once due to the pair comb case.

        frst_ft_terms = input_specs[0,:].copy()

        n_frst_ft_terms = frst_ft_terms.size

        # For the single pair case, the power spectrum is taken i.e.
        # its own variance.
        # For the multipair case, magnitude spectrum is taken. This is
        # sort of similar, I think.
        if n_frst_ft_terms == 1:
            input_spec_prod = np.prod(input_specs[1:,:] ** 2, axis=1)

        else:
            input_spec_prod = np.prod(input_specs[1:,:], axis=1)

        input_spec_cumsum = np.concatenate(
            (frst_ft_terms, input_spec_prod.cumsum()))

        if vtype == 'sim':
            norm_val = None
            sclrs = None
            frst_ft_terms = None

        elif vtype == 'ref':
            if n_frst_ft_terms == 1:
                norm_val = np.prod(
                    (input_specs[1:,:] ** 2).sum(axis=0) ** 2) ** 0.5

            else:
                norm_val = np.prod(
                    (input_specs[1:,:] ** n_frst_ft_terms
                    ).sum(axis=0)) ** (1 / n_frst_ft_terms)

            input_spec_cumsum[n_frst_ft_terms:] /= norm_val

            input_spec_cumsum[:n_frst_ft_terms] = 1

            # sclrs lets the first few long amplitudes into account much
            # better. These describe the direction i.e. Asymmetries.
            sclrs = 1.0 / np.arange(1.0, input_spec_cumsum.size + 1.0)
            sclrs[:n_frst_ft_terms] = 1.0  # input_spec_cumsum.size / n_frst_ft_terms

        else:
            raise NotImplementedError

        return (
            input_spec_cumsum,
            norm_val,
            sclrs,
            n_frst_ft_terms,
            frst_ft_terms)

