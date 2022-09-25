'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from ...misc import sci_round


class GTGAlgAutoObjWts:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self):

        return

    def _update_obj_wts(self):

        '''
        Less weights assigned to objective values that are bigger, relatively.

        Based on:
        (wts * means).sum() == means.sum()
        '''

        # Max seems to perform better than min and mean.
        means = np.abs(np.array(self._alg_wts_obj_raw)).max(axis=0)

        assert np.all(np.isfinite(means))

        sum_means = means.sum()

        wts = []
        for i in range(means.size):
            wt = sum_means / means[i]
            wts.append(wt)

        wts = np.array(wts)

        assert np.all(np.isfinite(wts))

        wts = (wts.size * wts) / wts.sum()

        wts **= self._sett_wts_obj_exp

        wts_sclr = sum_means / (means * wts).sum()

        wts *= wts_sclr

        assert np.isclose((wts * means).sum(), means.sum())

        wts = sci_round(wts)

        self._sett_wts_obj_wts = wts
        return

    def _set_auto_obj_wts(self, *args):

        '''
        Before the call to this method, the following methods must be there:
        - self._get_next_iter_vars
        - self._update_sim

        Your implementation here should do the following:

        - Set self._sett_wts_obj_wts to None.
        - Set self._alg_wts_obj_raw to [].
        - Set self._alg_wts_obj_search_flag.
        - In a for loop for self._sett_wts_obj_n_iters times:
            - Call self._get_next_iter_vars to get variables required
                to generate a new series.
            - Call self._update_sim with variables from previous step.
            - Call self._get_obj_ftn_val
        - Cast self._alg_wts_obj_raw to np.ndarray with dtype of np.float64.
        - Check that self._alg_wts_obj_raw is a 2D array.
        - Check that self._alg_wts_obj_raw has more than one row.
        - Unset self._alg_wts_obj_search_flag.
        - Call self._update_obj_wts.
        - Set self._alg_wts_obj_raw to None.
        '''

        _ = args

        raise NotImplementedError('Implement your own!')
        return
