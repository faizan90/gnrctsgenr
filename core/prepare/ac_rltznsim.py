'''
Created on Nov 16, 2021

@author: Faizan3800X-Uni
'''

from .aa_rltzngnrc import GTGPrepareRltznGnrc as GTGRG


class GTGPrepareRltznSim(GTGRG):

    def __init__(self):

        GTGRG.__init__(self)

        # Add var labs to _getdata in save.py if they need to be there.

        self.shape = None
        self.ft_best = None

        self.ref_sim_ft_corr = None
        self.sim_sim_ft_corr = None

        self.n_idxs_all_cts = None
        self.n_idxs_acpt_cts = None

        # Objective function variables.
        self.scorr_diffs = None
        self.asymm_1_diffs = None
        self.asymm_2_diffs = None
        self.ecop_dens_diffs = None
        self.ecop_etpy_diffs = None
        self.nth_ord_diffs = None
        self.pcorr_diffs = None

        self.mult_asymm_1_diffs = None
        self.mult_asymm_2_diffs = None
        self.mult_ecop_dens = None

        self.asymm_1_diffs_ft = None
        self.asymm_2_diffs_ft = None
        self.nth_ord_diffs_ft = None
        self.etpy_ft = None
        self.mult_asymm_1_cmpos_ft = None
        self.mult_asymm_2_cmpos_ft = None
        self.mult_etpy_cmpos_ft = None

        # QQ probs.
        self.scorr_qq_dict = None
        self.asymm_1_qq_dict = None
        self.asymm_2_qq_dict = None
        self.ecop_dens_qq_dict = None
        self.ecop_etpy_qq_dict = None
        self.nth_ords_qq_dict = None
        self.pcorr_qq_dict = None

        self.mult_asymm_1_qq_dict = None
        self.mult_asymm_2_qq_dict = None
        self.mult_ecop_dens_qq_dict = None  # TODO: This is not used anywhere.

        # Durations.
        self.cumm_call_durations = None
        self.cumm_n_calls = None

        # Optimization state and other variables.
        self.iter_ctr = None
        self.iters_wo_acpt = None
        self.tol = None
        self.temp = None
        self.stopp_criteria = None
        self.tols = None
        self.obj_vals_all = None
        self.acpts_rjts_all = None
        self.acpt_rates_all = None
        self.obj_vals_min = None
        self.temps = None
        self.acpt_rates_dfrntl = None
        self.ref_sim_ft_corr = None
        self.sim_sim_ft_corr = None
        self.obj_vals_all_indiv = None
        return