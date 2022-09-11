'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from time import asctime
from timeit import default_timer

import h5py
import numpy as np
from multiprocessing import Manager, Lock
from pathos.multiprocessing import ProcessPool

from ...misc import (
    print_sl, print_el, ret_mp_idxs, show_formatted_elapsed_time)


class GTGAlgorithm:

    '''The main annealing algorithm class.'''

    def __init__(self):

        return

    def simulate(self):

        '''Start the simulated annealing algorithm.'''

        assert self._alg_verify_flag, 'Call verify first!'

        beg_sim_tm = default_timer()

        self._init_output()

        if self._sett_auto_temp_set_flag:
            self._sett_misc_auto_init_temp_dir.mkdir(exist_ok=True)

            assert self._sett_misc_auto_init_temp_dir.exists(), (
                'Could not create auto_init_temp_dir!')

        beg_wts_tm = default_timer()

        self._update_wts()

        end_wts_tm = default_timer()

        if self._vb and ((end_wts_tm - beg_wts_tm) > 0.1):
            time_wts_str = show_formatted_elapsed_time(
                end_wts_tm - beg_wts_tm)

            print_sl()

            print(f'All weights\' computation took {time_wts_str}.')

            print_el()

        if self._sett_auto_temp_set_flag:
            self._search_init_temp()

        if self._vb:
            print_sl()

            temp = self._sett_ann_init_temp
            assert temp > self._alg_cnsts_almost_zero, (
                'Initial temperature almost zero!')

            ctr = 0
            while True:
                ctr += self._sett_ann_upt_evry_iter
                temp *= self._sett_ann_temp_red_rate

                if temp <= self._alg_cnsts_almost_zero:
                    break

            print(
                f'Maximum number of iterations possible with the set initial '
                f'temperature ({self._sett_ann_init_temp:5.3E}) are '
                f'{ctr:1.1E}.')

            if self._sett_ann_max_iters > ctr:
                print(
                    f'Info: set maximum number of iterations '
                    f'({self._sett_ann_max_iters:1.1E}) unreachable with '
                    f'this initial temperature!')

                self._sett_ann_max_iters = ctr

                print(
                    f'Set maximum number of iterations to: '
                    f'{self._sett_ann_max_iters:1.1E}')

            print_el()

        self._write_non_sim_data_to_h5()

        if self._vb:
            print_sl()

            print('Starting simulations...')

            print_el()

            print('\n')

        n_cpus = min(self._sett_misc_n_rltzns, self._sett_misc_n_cpus)

        if n_cpus > 1:

            # mp_idxs = ret_mp_idxs(self._sett_misc_n_rltzns, n_cpus)
            mp_idxs = np.arange(self._sett_misc_n_rltzns + 1)

            rltzns_gen = (
                (
                (mp_idxs[i], mp_idxs[i + 1]),
                )
                for i in range(mp_idxs.size - 1))

            self._lock = Manager().Lock()

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            list(mp_pool.uimap(self._simu_grp, rltzns_gen, chunksize=1))

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

        else:
            self._lock = Lock()

            self._simu_grp(((0, self._sett_misc_n_rltzns),))

            self._lock = None

        end_sim_tm = default_timer()

        if self._vb:
            print_sl()

            time_sim_str = show_formatted_elapsed_time(
                end_sim_tm - beg_sim_tm)

            print(f'Total simulation time was: {time_sim_str}!')

            print_el()

        self._alg_rltzns_gen_flag = True
        return

    def _init_output(self):

        if self._vb:
            print_sl()

            print('Initializing outputs file...')

        if not self._sett_misc_outs_dir.exists():
            self._sett_misc_outs_dir.mkdir(exist_ok=True)

        assert self._sett_misc_outs_dir.exists(), (
            'Could not create outputs_dir!')

        h5_path = self._sett_misc_outs_dir / self._save_h5_name

        # Create new / overwrite old.
        # It's simpler to do it here.
        h5_hdl = h5py.File(h5_path, mode='w', driver=None)
        h5_hdl.close()

        if self._vb:
            print('Initialized the outputs file.')

            print_el()
        return

    def _simu_grp(self, args):

        ((beg_rltzn_iter, end_rltzn_iter),
        ) = args

        beg_thread_tm = default_timer()

        for rltzn_iter in range(beg_rltzn_iter, end_rltzn_iter):

            self._reset_timers()

            self._gen_ref_aux_data()

            if self._vb:
                with self._lock:
                    print(
                        f'Started realization {rltzn_iter} on {asctime()}...')

            beg_rltzn_tm = default_timer()

            stopp_criteria = self._gen_gnrc_rltzn(
                (rltzn_iter, None))

            end_rltzn_tm = default_timer()

            if self._vb:
                with self._lock:

                    print('\n')

                    assert len(stopp_criteria) == len(
                        self._alg_cnsts_stp_crit_labs)

                    stopp_criteria_labels_rltzn = [
                        self._alg_cnsts_stp_crit_labs[i]
                        for i in range(len(stopp_criteria))
                        if not stopp_criteria[i]]

                    assert len(stopp_criteria_labels_rltzn), (
                        'No stopp_criteria!')

                    stopp_criteria_str = ' and '.join(
                        stopp_criteria_labels_rltzn)

                    time_rltzn_str = show_formatted_elapsed_time(
                        end_rltzn_tm - beg_rltzn_tm)

                    print(
                        f'Realization {rltzn_iter} took '
                        f'{time_rltzn_str} with stopp_criteria: '
                        f'{stopp_criteria_str}.')

            self._reset_timers()

        end_thread_tm = default_timer()

        if self._vb:

            time_thread_str = show_formatted_elapsed_time(
                end_thread_tm - beg_thread_tm)

            if (end_rltzn_iter - beg_rltzn_iter - 1):
                with self._lock: print(
                    f'Total thread time for realizations between '
                    f'{beg_rltzn_iter} and {end_rltzn_iter - 1} was '
                    f'{time_thread_str}.')
        return

    def verify(self):

        assert self._alg_cnsts_stp_crit_labs_flag, (
            'Stop criteria labels not set!')

        assert self._prep_verify_flag, 'Prepare in an unverified state!'

        if self._vb:
            print_sl()

            print(
                'Simulated annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    __verify = verify
