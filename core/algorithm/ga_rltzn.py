'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''


class GTGAlgRealization:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.

    For all search type algorithms, update _get_next_idxs accordingly
    to allow for full spectrum randomization during the search. The full
    spectrum randomization minimizes the variability of the objective function
    values, in case they start at an unwanted point.
    '''

    def __init__(self):

        return

    def _show_lag_nth_wts(self):

        if self._sett_obj_scorr_flag:
            print('wts_lag_scorr:', self._alg_wts_lag_scorr)

        if self._sett_obj_asymm_type_1_flag:
            print('wts_lag_asymm_1:', self._alg_wts_lag_asymm_1)

        if self._sett_obj_asymm_type_2_flag:
            print('wts_lag_asymm_2:', self._alg_wts_lag_asymm_2)

        if self._sett_obj_ecop_dens_flag:
            print('wts_lag_ecop_dens:', self._alg_wts_lag_ecop_dens)

        if self._sett_obj_ecop_etpy_flag:
            print('wts_lag_ecop_etpy:', self._alg_wts_lag_ecop_etpy)

        if self._sett_obj_nth_ord_diffs_flag:
            print('wts_nth_order:', self._alg_wts_nth_order)

        if self._sett_obj_pcorr_flag:
            print('wts_lag_pcorr:', self._alg_wts_lag_pcorr)

        if self._sett_obj_asymm_type_1_ft_flag:
            print('wts_lag_asymm_1_ft:', self._alg_wts_lag_asymm_1_ft)

        if self._sett_obj_asymm_type_2_ft_flag:
            print('wts_lag_asymm_2_ft:', self._alg_wts_lag_asymm_2_ft)

        if self._sett_obj_nth_ord_diffs_ft_flag:
            print('wts_nth_order_ft:', self._alg_wts_nth_order_ft)

        if self._sett_obj_etpy_ft_flag:
            print('wts_lag_etpy_ft:', self._alg_wts_lag_etpy_ft)

        return

    def _show_label_wts(self):

        if self._sett_obj_scorr_flag:
            print('wts_label_scorr:', self._alg_wts_label_scorr)

        if self._sett_obj_asymm_type_1_flag:
            print('wts_label_asymm_1:', self._alg_wts_label_asymm_1)

        if self._sett_obj_asymm_type_2_flag:
            print('wts_label_asymm_2:', self._alg_wts_label_asymm_2)

        if self._sett_obj_ecop_dens_flag:
            print(
                'wts_label_ecop_dens:', self._alg_wts_label_ecop_dens)

        if self._sett_obj_ecop_etpy_flag:
            print(
                'wts_label_ecop_etpy:', self._alg_wts_label_ecop_etpy)

        if self._sett_obj_nth_ord_diffs_flag:
            print(
                'wts_label_nth_order:', self._alg_wts_label_nth_order)

        if self._sett_obj_pcorr_flag:
            print('wts_label_pcorr:', self._alg_wts_label_pcorr)

        if self._sett_obj_asymm_type_1_ft_flag:
            print(
                'wts_label_asymm_1_ft:',
                self._alg_wts_label_asymm_1_ft)

        if self._sett_obj_asymm_type_2_ft_flag:
            print(
                'wts_label_asymm_2_ft:',
                self._alg_wts_label_asymm_2_ft)

        if self._sett_obj_nth_ord_diffs_ft_flag:
            print(
                'wts_label_nth_order_ft:',
                self._alg_wts_label_nth_order_ft)

        if self._sett_obj_etpy_ft_flag:
            print(
                'wts_label_etpy_ft:',
                self._alg_wts_label_etpy_ft)

        return

    def _show_obj_wts(self):

        _obj_labs = self._sett_obj_flag_labels[
            self._sett_obj_flag_vals]

        print(
            'Obj. wts.:',
            [f'{_obj_labs[i]}: {self._sett_wts_obj_wts[i]:2.2E}'
             for i in range(len(_obj_labs))])

        return

    def _update_wts(self, *args):

        '''
        Make call(s) to required method(s) that compute the
        lag, label and obj weigths.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    def _load_snapshot(self):

        # NOTE: Synchronize changes with _update_snapshot.

        (self._rs.scorrs,
         self._rs.asymms_1,
         self._rs.asymms_2,
         self._rs.ecop_dens,
         self._rs.ecop_etpy,
         self._rs.pcorrs,
         self._rs.nths,
         self._rs.data_ft,
         self._rs.probs_ft,

         self._rs.scorr_diffs,
         self._rs.asymm_1_diffs,
         self._rs.asymm_2_diffs,
         self._rs.ecop_dens_diffs,
         self._rs.ecop_etpy_diffs,
         self._rs.nth_ord_diffs,
         self._rs.pcorr_diffs,

         self._rs.asymm_1_diffs_ft,
         self._rs.asymm_2_diffs_ft,
         self._rs.nth_ord_diffs_ft,
         self._rs.etpy_ft,

         self._rs.mult_asymm_1_diffs,
         self._rs.mult_asymm_2_diffs,
         self._rs.mult_ecop_dens,
         self._rs.mult_asymm_1_cmpos_ft,
         self._rs.mult_asymm_2_cmpos_ft,

         self._rs.scorrs_ms,
         self._rs.ecop_etpy_ms,

         self._rs.data_ms_ft,
         self._rs.probs_ms_ft,

         self._rs.data_ms_pair_ft,
         self._rs.probs_ms_pair_ft,

        ) = self._alg_snapshot['obj_vars']

        self._rs.data = self._alg_snapshot['data']
        self._rs.probs = self._alg_snapshot['probs']

        self._rs.data_ft_coeffs = self._alg_snapshot['data_ft_coeffs']

        self._rs.data_ft_coeffs_mags = self._alg_snapshot[
            'data_ft_coeffs_mags']

        self._rs.data_ft_coeffs_phss = self._alg_snapshot[
            'data_ft_coeffs_phss']

        self._rs.probs_ft_coeffs = self._alg_snapshot['probs_ft_coeffs']

        self._rs.probs_ft_coeffs_mags = self._alg_snapshot[
            'probs_ft_coeffs_mags']

        self._rs.probs_ft_coeffs_phss = self._alg_snapshot[
            'probs_ft_coeffs_phss']
        return

    def _update_snapshot(self):

        # NOTE: Synchronize changes with _load_snapshot.

        obj_vars = (
            self._rs.scorrs,
            self._rs.asymms_1,
            self._rs.asymms_2,
            self._rs.ecop_dens,
            self._rs.ecop_etpy,
            self._rs.pcorrs,
            self._rs.nths,
            self._rs.data_ft,
            self._rs.probs_ft,

            self._rs.scorr_diffs,
            self._rs.asymm_1_diffs,
            self._rs.asymm_2_diffs,
            self._rs.ecop_dens_diffs,
            self._rs.ecop_etpy_diffs,
            self._rs.nth_ord_diffs,
            self._rs.pcorr_diffs,

            self._rs.asymm_1_diffs_ft,
            self._rs.asymm_2_diffs_ft,
            self._rs.nth_ord_diffs_ft,
            self._rs.etpy_ft,

            self._rs.mult_asymm_1_diffs,
            self._rs.mult_asymm_2_diffs,
            self._rs.mult_ecop_dens,
            self._rs.mult_asymm_1_cmpos_ft,
            self._rs.mult_asymm_2_cmpos_ft,

            self._rs.scorrs_ms,
            self._rs.ecop_etpy_ms,

            self._rs.data_ms_ft,
            self._rs.probs_ms_ft,

            self._rs.data_ms_pair_ft,
            self._rs.probs_ms_pair_ft,
            )

        self._alg_snapshot = {
            'obj_vars': obj_vars,
            'data': self._rs.data,
            'probs': self._rs.probs,
            'data_ft_coeffs': self._rs.data_ft_coeffs,
            'data_ft_coeffs_mags': self._rs.data_ft_coeffs_mags,
            'data_ft_coeffs_phss': self._rs.data_ft_coeffs_phss,
            'probs_ft_coeffs': self._rs.probs_ft_coeffs,
            'probs_ft_coeffs_mags': self._rs.probs_ft_coeffs_mags,
            'probs_ft_coeffs_phss': self._rs.probs_ft_coeffs_phss,
            }

        return

    def _show_rltzn_situ(self, *args):

        '''
        Implement information to show ongoing situation of each
        thread after 10 percent of iterations.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    def _get_stopp_criteria(self, *args):

        '''
        Implement stopping criteria that returns a tuple of boolean values.
        If any value in the tuple is False, the optimization stops.
        This does not apply to various lag, label and obj weights'
        computation or the automatic temperature detection realizations.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    # TODO: _timer_wrap this later.
    def _get_next_iter_vars(self, *args):

        '''
        Should generate required information required by update_sim.

        e.g. Old and new indices or phases or noise.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    def _update_sim_no_prms(self, *args):

        '''
        Update simulation variables without any arguments.

        Implement updating of self._rs.data and self._rs.probs
        and then calling self._update_obj_vars('sim').
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    # TODO: _timer_wrap this later.
    def _update_sim(self, *args):

        '''
        Perturb a series by adding noise or phase changes or whatever here.
        The state of load_snapshot_flag in the arguments determines
        whether to revert to old state or update simulation.
        - if load_snapshot_flag is True then call self._load_snapshot.
        - if load_snapshot_flag is False then call self._update_sim_no_prms.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return

    def _gen_gnrc_rltzn(self, *args):

        '''
        The main algorithm that generates a time series realization.

        As arguments, it takes the index of a simulation and the initial
        annealing temperature.

        There are many variables to take of here. This is best illustrated
        by code rather than pseudocode.

        Take a look at the _gen_gnrc_rltzn in example_implementation directory
        to get a overall picture of what should be there.
        There the phase annealing algorithm is implemented.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return
