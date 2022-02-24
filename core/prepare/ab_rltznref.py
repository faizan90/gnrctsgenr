'''
Created on Nov 16, 2021

@author: Faizan3800X-Uni
'''

from .aa_rltzngnrc import GTGPrepareRltznGnrc as GTGRG


class GTGPrepareRltznRef(GTGRG):

    def __init__(self):

        GTGRG.__init__(self)

        self.label = 'ref'

        self.ft_cumm_corr = None  # Of the transformed series.
        self.probs_srtd = None
        self.data_ft_norm_vals = None  # To norm the sims with.
        self.data_tfm = None  # Transformed data.
        self.probs_ft_norm_vals = None  # To norm the sims with.

        # Reference data for objective functions.
        self.scorr_diffs_cdfs_dict = None  # Distributed bivariate.
        self.asymm_1_diffs_cdfs_dict = None  # Distributed bivariate.
        self.asymm_2_diffs_cdfs_dict = None  # Distributed bivariate.
        self.ecop_dens_diffs_cdfs_dict = None  # Distributed bivariate.
        self.ecop_etpy_diffs_cdfs_dict = None  # Distributed bivariate.
        self.nth_ord_diffs_cdfs_dict = None  # Distributed.
        self.pcorr_diffs_cdfs_dict = None  # Distributed bivariate.
        self.cos_sin_cdfs_dict = None  # Distributed.

        self.mult_asymm_1_diffs_cdfs_dict = None  # Distributed bivariate.
        self.mult_asymm_2_diffs_cdfs_dict = None  # Distributed bivariate.
        self.mult_ecop_dens_cdfs_dict = None  # Distributed bivariate.

        self.scorr_qq_dict = None  # Distributed bivariate.
        self.asymm_1_qq_dict = None  # Distributed bivariate.
        self.asymm_2_qq_dict = None  # Distributed bivariate.
        self.ecop_dens_qq_dict = None  # Distributed bivariate.
        self.ecop_etpy_qq_dict = None  # Distributed bivariate.
        self.nth_ord_qq_dict = None  # Distributed.
        self.pcorr_qq_dict = None  # Distributed bivariate.

        self.mult_asymm_1_qq_dict = None  # Distributed bivariate.
        self.mult_asymm_2_qq_dict = None  # Distributed bivariate.
        self.mult_etpy_dens_qq_dict = None  # Distributed bivariate.

        self.asymm_1_diffs_ft_dict = None  # Distributed bivariate.
        self.asymm_2_diffs_ft_dict = None  # Distributed bivariate.
        self.nth_ord_diffs_ft_dict = None  # Distributed.
        self.etpy_ft_dict = None  # Distributed bivariate.
        self.mult_asymm_1_cmpos_ft_dict = None  # Distributed bivariate.
        self.mult_asymm_2_cmpos_ft_dict = None  # Distributed bivariate.
        self.mult_etpy_cmpos_ft_dict = None  # Distributed bivariate.

        self.data_ms_ft_norm_val = None  # Multivariate norming value for max_corr.
        self.probs_ms_ft_norm_val = None  # Multivariate norming value for max_corr.

        self.data_ms_pair_ft_norm_vals = None  # Bivariate norming value for pairwise FT.
        self.probs_ms_pair_ft_norm_vals = None  # Bivariate norming value for pairwise FT.
        return
