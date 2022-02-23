'''
Created on Nov 16, 2021

@author: Faizan3800X-Uni
'''


class GTGPrepareRltznGnrc:

    def __init__(self):

        # Add var labs to _getdata in save.py if they need to be there.

        self.label = None
        self.probs = None
        self.ft = None  # FT coefficients of the transformed series.
        self.phs_spec = None  # Of the transformed series.
        self.mag_spec = None  # Of the transformed series.
        self.scorrs = None  # Lumped bivariate.
        self.asymms_1 = None  # Lumped bivariate.
        self.asymms_2 = None  # Lumped bivariate.
        self.ecop_dens = None  # Lumped bivariate.
        self.ecop_etpy = None  # Lumped bivariate.
        self.data = None  # Original (reshuffled for sims) data.
        self.pcorrs = None  # Lumped bivariate.
        self.nths = None  # Lumped nth order sums.
        self.data_ft = None  # Actually, cumm_corr_spec.
        self.probs_ft = None  # Actually, cumm_corr_spec.
        self.scorrs_ms = None  # Multivariate Spearman corr.
        self.ecop_etpy_ms = None  # Lumped, on a multivariate E. copula.
        self.data_ms_ft = None  # Actually, cumm_max_corr_spec.
        self.probs_ms_ft = None  # Actually, cumm_max_corr_spec.

        self.skip_io_vars = []  # Not to write in the main save function.
        return
