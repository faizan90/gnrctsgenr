'''
Created on Nov 17, 2021

@author: Faizan3800X-Uni
'''


class GTGPrepareBase:

    def __init__(self):

        self._data_tfm_type = 'probs'
        self._data_tfm_types = (
            'log_data', 'probs', 'data', 'probs_sqrt', 'norm')

        # Flags.
        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False
        self._prep_prep_flag = False
        self._prep_verify_flag = False

        # Validation steps.
        self._prep_vld_flag = False
        return
