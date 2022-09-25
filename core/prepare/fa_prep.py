'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np
from scipy.stats import norm

from ...misc import (
    print_sl,
    print_el,
    )


class GTGPrepare:

    '''Prepare derived variables required here.'''

    def __init__(self):

        return

    def _get_data_tfm(self, data, probs):

        assert self._sett_data_tfm_type in self._sett_data_tfm_types, (
            f'Unknown data transform string {self._sett_data_tfm_type}!')

        if self._sett_data_tfm_type == 'log_data':
            data_tfm = np.log(data)

        elif self._sett_data_tfm_type == 'probs':
            data_tfm = probs.copy()

        elif self._sett_data_tfm_type == 'data':
            data_tfm = data.copy()

        elif self._sett_data_tfm_type == 'probs_sqrt':
            data_tfm = probs ** 0.5

        elif self._sett_data_tfm_type == 'norm':
            data_tfm = norm.ppf(probs)

        else:
            raise NotImplementedError()

        assert np.all(np.isfinite(data_tfm)), 'Invalid values in data_tfm!'

        return data_tfm.copy(order='f')

    def _gen_ref_aux_data_gnrc(self):

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        probs = self._get_probs(self._data_ref_rltzn, False)

        self._rr.data_tfm = self._get_data_tfm(self._data_ref_rltzn, probs)

        ft = np.fft.rfft(self._rr.data_tfm, axis=0)

        self._rr.data = self._data_ref_rltzn.copy(order='f')

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'
        assert np.all(np.isfinite(mag_spec)), 'Invalid values in mag_spec!'

        self._rr.probs = probs
        self._rr.probs_srtd = np.sort(probs, axis=0)

        self._rr.ft = ft
        self._rr.phs_spec = phs_spec
        self._rr.mag_spec = mag_spec

        # if any([self._sett_obj_match_data_ft_flag,
        #         self._sett_obj_match_data_ms_ft_flag,
        #         self._sett_obj_match_data_ms_pair_ft_flag]):

        self._rr.data_ft_coeffs = np.fft.rfft(self._data_ref_rltzn, axis=0)
        self._rr.data_ft_coeffs_mags = np.abs(self._rr.data_ft_coeffs)
        self._rr.data_ft_coeffs_phss = np.angle(self._rr.data_ft_coeffs)

        # if any([self._sett_obj_match_probs_ft_flag,
        #         self._sett_obj_match_probs_ms_ft_flag,
        #         self._sett_obj_match_probs_ms_pair_ft_flag]):

        self._rr.probs_ft_coeffs = np.fft.rfft(self._rr.probs, axis=0)
        self._rr.probs_ft_coeffs_mags = np.abs(self._rr.probs_ft_coeffs)
        self._rr.probs_ft_coeffs_phss = np.angle(self._rr.probs_ft_coeffs)

        self._update_obj_vars('ref')

        if self._sett_obj_use_obj_dist_flag:
            if self._sett_obj_scorr_flag:
                self._rr.scorr_diffs_cdfs_dict = (
                    self._get_scorr_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_1_flag:
                self._rr.asymm_1_diffs_cdfs_dict = (
                    self._get_asymm_1_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_2_flag:
                self._rr.asymm_2_diffs_cdfs_dict = (
                    self._get_asymm_2_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_ecop_dens_flag:
                self._rr.ecop_dens_diffs_cdfs_dict = (
                    self._get_ecop_dens_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_ecop_etpy_flag:
                self._rr.ecop_etpy_diffs_cdfs_dict = (
                    self._get_ecop_etpy_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_nth_ord_diffs_flag:
                self._rr.nth_ord_diffs_cdfs_dict = (
                    self._get_nth_ord_diffs_cdfs_dict(
                        self._rr.data, self._sett_obj_nth_ords_vld))

            if self._sett_obj_pcorr_flag:
                self._rr.pcorr_diffs_cdfs_dict = (
                    self._get_pcorr_diffs_cdfs_dict(self._rr.data))

        if self._sett_obj_asymm_type_1_ft_flag:
            self._rr.asymm_1_diffs_ft_dict = (
                self._get_asymm_1_diffs_ft_dict(self._rr.probs))

        if self._sett_obj_asymm_type_2_ft_flag:
            self._rr.asymm_2_diffs_ft_dict = (
                self._get_asymm_2_diffs_ft_dict(self._rr.probs))

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._rr.nth_ord_diffs_ft_dict = (
                self._get_nth_ord_diffs_ft_dict(
                    self._rr.data, self._sett_obj_nth_ords_vld))

        if self._sett_obj_etpy_ft_flag:
            self._rr.etpy_ft_dict = (
                self._get_etpy_ft_dict(self._rr.probs))

        if self._data_ref_n_labels > 1:
            if self._sett_obj_asymm_type_1_ms_flag:
                self._rr.mult_asymm_1_diffs_cdfs_dict = (
                    self._get_mult_asymm_1_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_2_ms_flag:
                self._rr.mult_asymm_2_diffs_cdfs_dict = (
                    self._get_mult_asymm_2_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_ecop_dens_ms_flag:
                self._rr.mult_ecop_dens_cdfs_dict = (
                    self._get_mult_ecop_dens_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_1_ms_ft_flag:
                self._rr.mult_asymm_1_cmpos_ft_dict = (
                    self._get_mult_asymm_1_cmpos_ft(self._rr.probs, 'ref'))

            if self._sett_obj_asymm_type_2_ms_ft_flag:
                self._rr.mult_asymm_2_cmpos_ft_dict = (
                    self._get_mult_asymm_2_cmpos_ft(self._rr.probs, 'ref'))

            if self._sett_obj_etpy_ms_ft_flag:
                self._rr.mult_etpy_cmpos_ft_dict = (
                    self._get_mult_etpy_cmpos_ft(self._rr.probs, 'ref'))

#             self._get_mult_scorr_cmpos_ft(self._rr.probs, 'ref')

        if self._sett_obj_cos_sin_dist_flag:
            self._rr.cos_sin_cdfs_dict = self._get_cos_sin_cdfs_dict(
                self._rr.ft)

        return

    def _gen_ref_aux_data(self):

        # Call _gen_ref_aux_data_gnrc in own implmentation.

        raise NotImplementedError('Implement your own!')

        return

    def _gen_sim_aux_data(self):

        raise NotImplementedError('Implement your own!')

        return

    def prepare(self):

        '''Generate data required before annealing starts.'''

        assert self._sett_verify_flag, 'Settings in an unverfied state!'

        self._gen_ref_aux_data()
        assert self._prep_ref_aux_flag, (
            'Apparently, _gen_ref_aux_data did not finish as expected!')

        self._gen_sim_aux_data()
        assert self._prep_sim_aux_flag, (
            'Apparently, _gen_sim_aux_data did not finish as expected!')

        self._prep_prep_flag = True
        return

    def verify(self):

        # assert self._prep_prep_flag, 'Call prepare first!'

        if self._vb:
            print_sl()

            print(f'Preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify
