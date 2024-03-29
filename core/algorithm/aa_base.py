'''
Created on Nov 17, 2021

@author: Faizan3800X-Uni
'''

from ...misc import print_sl, print_el


class GTGAlgBase:

    def __init__(self):

        self._lock = None

        self._alg_rltzn_iter = None

        # Snapshot.
        self._alg_snapshot = None

        # Lag/Nth  weights.
        self._alg_wts_lag_nth_search_flag = False
        self._alg_wts_lag_scorr = None
        self._alg_wts_lag_asymm_1 = None
        self._alg_wts_lag_asymm_2 = None
        self._alg_wts_lag_ecop_dens = None
        self._alg_wts_lag_ecop_etpy = None
        self._alg_wts_nth_order = None
        self._alg_wts_lag_pcorr = None
        self._alg_wts_lag_asymm_1_ft = None
        self._alg_wts_lag_asymm_2_ft = None
        self._alg_wts_nth_order_ft = None
        self._alg_wts_lag_etpy_ft = None

        # Label  weights.
        self._alg_wts_label_search_flag = False
        self._alg_wts_label_scorr = None
        self._alg_wts_label_asymm_1 = None
        self._alg_wts_label_asymm_2 = None
        self._alg_wts_label_ecop_dens = None
        self._alg_wts_label_ecop_etpy = None
        self._alg_wts_label_nth_order = None
        self._alg_wts_label_pcorr = None
        self._alg_wts_label_asymm_1_ft = None
        self._alg_wts_label_asymm_2_ft = None
        self._alg_wts_label_nth_order_ft = None
        self._alg_wts_label_etpy_ft = None
        self._alg_wts_label_cos_sin_dist = None
        self._alg_wts_label_data_ft = None
        self._alg_wts_label_probs_ft = None

        # Obj wts.
        self._sett_wts_obj_wts = None
        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False

        # Stopping criteria labels.
        self._alg_cnsts_stp_crit_labs = None

        # Closest value that is less than or equal to be taken as a zero.
        self._alg_cnsts_almost_zero = 1e-15

        # Exponent to take before computing sum of differences in
        # objective functions. Must be an even value.
        self._alg_cnsts_diffs_exp = 2.0

        # Limiting values of minimum and maximum probabilites in the
        # objective functions when simulation probabilities are computed
        # inversly. This limits the cases when the extrapolation goes way
        # below or beyond, 0 and 1 respectively.
        self._alg_cnsts_min_prob_val = -0.1
        self._alg_cnsts_max_prob_val = +1.1

        # A flag to tell if to use non-exceedence probabilities for computing
        # the objective values or the histogram. This is used along with
        # other criteria. So, the effect can be different when the flag is
        # set or unset.
        self._alg_cnsts_lag_wts_overall_err_flag = True

        # Flags.
        self._alg_cnsts_stp_crit_labs_flag = False
        self._alg_rltzns_gen_flag = False
        self._alg_force_acpt_flag = False
        self._alg_done_opt_flag = False
        self._alg_ann_runn_auto_init_temp_search_flag = False
        self._alg_verify_flag = False
        return

    def set_stop_criteria_labels(self, stop_criteria_labels):

        '''
        Set the labels for the stopping criteria. This information is
        used when showing stopping criterium that caused the annealing to
        stop.

        Parameters
        ----------
        stop_criteria_labels : list or tuple of strings
            These should be the same length as the number of variabels
            tested in the _get_stopp_criteria method. It cannot be emtpy.
            The labels must have a length of at least one.
        '''

        if self._vb:
            print_sl()

            print('Setting algortihm base settings...\n')

        assert isinstance(stop_criteria_labels, (list, tuple)), (
            'stop_criteria_labels not a list or tuple object!')

        assert len(stop_criteria_labels), 'Empty stop_criteria_labels!'

        assert all([isinstance(x, str) for x in stop_criteria_labels]), (
            'Invalid data types inside of stop_criteria_labels!')

        assert all([len(x) for x in stop_criteria_labels]), (
            'Empty strings inside of stop_criteria_labels!')

        self._alg_cnsts_stp_crit_labs = tuple(stop_criteria_labels)

        if self._vb:
            print(
                'Algorithm stop criteria labels:')

            for stop_crit_lab in self._alg_cnsts_stp_crit_labs:
                print('-', stop_crit_lab)

            print_el()

        self._alg_cnsts_stp_crit_labs_flag = True
        return
