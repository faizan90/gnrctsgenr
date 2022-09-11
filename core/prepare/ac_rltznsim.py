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

        self.ref_sim_ft_corr = None  # Transformed data.
        self.sim_sim_ft_corr = None  # Transformed data.

        # Objective function variables.
        self.scorr_diffs = None  # Distributed bivariate.
        self.asymm_1_diffs = None  # Distributed bivariate.
        self.asymm_2_diffs = None  # Distributed bivariate.
        self.ecop_dens_diffs = None  # Distributed bivariate.
        self.ecop_etpy_diffs = None  # Distributed bivariate.
        self.nth_ord_diffs = None  # Distributed.
        self.pcorr_diffs = None  # Distributed bivariate.

        self.mult_asymm_1_diffs = None  # Distributed bivariate.
        self.mult_asymm_2_diffs = None  # Distributed bivariate.
        self.mult_ecop_dens = None  # Distributed bivariate.

        self.asymm_1_diffs_ft = None  # Distributed bivariate.
        self.asymm_2_diffs_ft = None  # Distributed bivariate.
        self.nth_ord_diffs_ft = None  # Distributed.
        self.etpy_ft = None  # Distributed bivariate.
        self.mult_asymm_1_cmpos_ft = None  # Distributed bivariate.
        self.mult_asymm_2_cmpos_ft = None  # Distributed bivariate.
        self.mult_etpy_cmpos_ft = None  # Distributed bivariate.

        # QQ probs.
        self.scorr_qq_dict = None  # Distributed bivariate.
        self.asymm_1_qq_dict = None  # Distributed bivariate.
        self.asymm_2_qq_dict = None  # Distributed bivariate.
        self.ecop_dens_qq_dict = None  # Distributed bivariate.
        self.ecop_etpy_qq_dict = None  # Distributed bivariate.
        self.nth_ords_qq_dict = None  # Distributed.
        self.pcorr_qq_dict = None  # Distributed bivariate.

        self.mult_asymm_1_qq_dict = None  # Distributed bivariate.
        self.mult_asymm_2_qq_dict = None  # Distributed bivariate.
        self.mult_ecop_dens_qq_dict = None  # TODO: This is not used anywhere.

        # Durations.
        self.cumm_call_durations = None  # Time spent in various methods.
        self.cumm_n_calls = None  # Number of times various methods are called.

        # Optimization state and other variables.
        self.iter_ctr = None  # Iteration counter.
        self.iters_wo_acpt = None  # Continuous iterations without acceptance.
        self.tol = None  # Iteration objective function value tolerance.
        self.temp = None  # Iteration temperature value.
        self.stopp_criteria = None  # State of stopping criteria.
        self.tols = None  # Running objective function values tolerance.
        self.obj_vals_all = None  # All objective function values.
        self.acpts_rjts_all = None  # All acceptance / rejection values.
        self.acpt_rates_all = None  # Running acceptance rate.
        self.obj_vals_min = None  # Running minimum objective function values.
        self.temps = None  # All temperatures.
        self.acpt_rates_dfrntl = None  # Acpt. rate of past few iterations.
        self.obj_vals_all_indiv = None  # All obj. ftn. values.
        return
