'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import time
from timeit import default_timer
from multiprocessing import Manager, Lock

import numpy as np
from pathos.multiprocessing import ProcessPool

from ..aa_base import GTGBase
from ...misc import print_sl, print_el, show_formatted_elapsed_time


class GTGAlgTemperature:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self):

        return

    def _get_acpt_rate_and_temp(self, args):

        attempt, init_temp, que_res, que_active_threads = args

        rltzn_args = (
            attempt,
            init_temp,
            )

        rltzn = self._gen_gnrc_rltzn(rltzn_args)

        if self._vb:
            with self._lock:
                print(
                    f'Acceptance rate and temperature for '
                    f'attempt {attempt}: {rltzn[0]:5.2%}, '
                    f'{rltzn[1]:0.3E}')

        if que_res is not None:
            que_res.put(rltzn)

        if que_active_threads is not None:
            que_active_threads.get()

        return rltzn

    def _get_auto_init_temp_and_interped_temps(self, acpt_rates_temps):

        '''
        First column is the acceptance rate, second is the temperature.
        '''

        if False:
            # This could cause trouble to get enough points.
            # First trying this. Fewer points near the required value result
            # in a better fit.
            min_acpt = max(
                self._sett_ann_auto_init_temp_acpt_min_bd_lo,
                self._sett_ann_auto_init_temp_trgt_acpt_rate - 0.15)

            max_acpt = min(
                self._sett_ann_auto_init_temp_acpt_max_bd_hi,
                self._sett_ann_auto_init_temp_trgt_acpt_rate + 0.05)

            keep_idxs = (
                (acpt_rates_temps[:, 0] > min_acpt) &
                (acpt_rates_temps[:, 0] < max_acpt)
                )

        else:
            # The upper one resulted in too few points.
            # This one allows for some more.
            best_app_idx = np.argmin(
                ((acpt_rates_temps[:, 0] -
                  self._sett_ann_auto_init_temp_trgt_acpt_rate) ** 2))

            keep_idxs = np.arange(
                max(0, best_app_idx - 5),
                min(best_app_idx + 4, acpt_rates_temps.shape[0]))

        n_keep_pts = keep_idxs.sum()

        if n_keep_pts < self._sett_ann_auto_init_temp_acpt_polyfit_n_pts:
            # If points taken above are not enough then take the maximum
            # allowed here. If this doesn't work out then an error is
            # raised.
            keep_idxs = (
                (acpt_rates_temps[:, 0] >
                 self._sett_ann_auto_init_temp_acpt_min_bd_lo) &
                (acpt_rates_temps[:, 0] <
                 self._sett_ann_auto_init_temp_acpt_max_bd_hi)
                )

        # Sanity check.
        assert keep_idxs.sum() >= 2

        assert (
            keep_idxs.sum() >=
            self._sett_ann_auto_init_temp_acpt_polyfit_n_pts), (
            f'Not enough usable points (n={keep_idxs.sum()}) for fitting a '
            f'curve to acceptance rates and temperatures!\n'
            f'Acceptance rates\' and temperatures\' matrix:\n'
            f'{acpt_rates_temps}')

        # acpt_rates_temps = acpt_rates_temps[keep_idxs,:].copy()

        # Odd values are bad. High values result in an unstable fit.
        # Choosing few values for the fit and poly_deg of 2 results
        # in a stable fit.
        poly_deg = 2
        if keep_idxs.sum() < poly_deg:
            poly_deg = keep_idxs.sum()

        if (poly_deg % 2) and (poly_deg > 2):
            poly_deg -= 1

        poly_coeffs = np.polyfit(
            acpt_rates_temps[keep_idxs, 0],
            acpt_rates_temps[keep_idxs, 1],
            deg=poly_deg)

        poly_ftn = np.poly1d(poly_coeffs)

        init_temp = poly_ftn(self._sett_ann_auto_init_temp_trgt_acpt_rate)

        interp_arr = np.empty((300, 2), dtype=float)

        interp_arr[:, 0] = np.linspace(
            acpt_rates_temps[0, 0], acpt_rates_temps[-1, 0], 300)

        interp_arr[:, 1] = poly_ftn(interp_arr[:, 0])

        return init_temp, interp_arr

    def _plot_acpt_rate_temps(
            self,
            interp_acpt_rates_temps,
            acpt_rates_temps,
            ann_init_temp):

        # The import has to be kept here. Putting it at the top created
        # strange crashes.

        import matplotlib.pyplot as plt
        from adjustText import adjust_text

        plt.figure(figsize=(10, 10))

        plt.plot(
            interp_acpt_rates_temps[:, 1],
            interp_acpt_rates_temps[:, 0],
            alpha=0.75,
            c='C0',
            lw=2,
            label='fitted',
            zorder=1)

        plt.scatter(
            acpt_rates_temps[:, 1],
            acpt_rates_temps[:, 0],
            alpha=0.75,
            c='C0',
            label='simulated',
            zorder=2)

        plt.vlines(
            ann_init_temp,
            0,
            self._sett_ann_auto_init_temp_trgt_acpt_rate,
            alpha=0.5,
            ls='--',
            lw=1,
            color='k',
            zorder=3)

        plt.hlines(
            self._sett_ann_auto_init_temp_trgt_acpt_rate,
            0,
            ann_init_temp,
            alpha=0.5,
            ls='--',
            lw=1,
            color='k',
            zorder=3)

        plt.scatter(
            [ann_init_temp],
            [self._sett_ann_auto_init_temp_trgt_acpt_rate],
            alpha=0.75,
            c='k',
            label='selected',
            zorder=4)

        ptexts = []
        ptext = plt.text(
            ann_init_temp,
            self._sett_ann_auto_init_temp_trgt_acpt_rate,
            f'({ann_init_temp:1.2E}, '
            f'{self._sett_ann_auto_init_temp_trgt_acpt_rate:.1%})',
            color='k',
            alpha=0.90,
            zorder=5)

        ptexts.append(ptext)

        adjust_text(ptexts, only_move={'points': 'y', 'text': 'y'})

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Temperature')
        plt.ylabel('Acceptance rate')

        plt.ylim(-0.05, +1.05)

        plt.xscale('log')

        out_fig_path = (
            self._sett_misc_auto_init_temp_dir / f'init_temps__acpt_rates.png')

        plt.savefig(str(out_fig_path), bbox_inches='tight')

        plt.close()
        return

    # TODO: _timer_wrap this later.
    @GTGBase._timer_wrap
    def _search_init_temp(self):

        beg_tm = default_timer()

        if self._vb:
            print_sl()
            print('Searching for initialization temperature...')
            print_el()

        self._alg_ann_runn_auto_init_temp_search_flag = True

        init_temps = [self._sett_ann_auto_init_temp_temp_bd_lo]

        while init_temps[-1] < self._sett_ann_auto_init_temp_temp_bd_hi:

            init_temps.append(
                init_temps[-1] * self._sett_ann_auto_init_temp_ramp_rate)

        if init_temps[-1] > self._sett_ann_auto_init_temp_temp_bd_hi:
            init_temps[-1] = self._sett_ann_auto_init_temp_temp_bd_hi

        n_init_temps = len(init_temps)

        if self._vb:
            print(f'Total possible attempts: {n_init_temps}')

            print_sl()

        assert (n_init_temps >=
                self._sett_ann_auto_init_temp_acpt_polyfit_n_pts), (
            'Not enough initial temperature detection iteration!')

        search_attempts = 0
        acpt_rates_temps = []

        n_cpus = min(n_init_temps, self._sett_misc_n_cpus)

        if n_cpus > 1:

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            mngr = Manager()

            self._lock = mngr.Lock()

            que_res = mngr.Queue(n_init_temps)
            que_active_threads = mngr.Queue(n_init_temps)

            keep_going_flag = True
            keep_searching_flag = True

            while keep_going_flag:

                while ((keep_searching_flag) and
                       (que_active_threads.qsize() <= n_cpus)):

                    mp_pool.uimap(
                        self._get_acpt_rate_and_temp,
                        ((search_attempts,
                          init_temps[search_attempts],
                          que_res,
                          que_active_threads),),
                        chunksize=1)

                    que_active_threads.put(search_attempts)

                    search_attempts += 1

                    if search_attempts == n_init_temps:

                        keep_searching_flag = False
                        break

                time.sleep(0.1)

                while que_res.qsize():
                    acpt_rates_temps.append(que_res.get_nowait())

                    if (acpt_rates_temps[-1][0] >=
                        self._sett_ann_auto_init_temp_acpt_max_bd_hi):

                        keep_searching_flag = False

                if ((not keep_searching_flag) and
                    (search_attempts == len(acpt_rates_temps))):

                    keep_going_flag = False

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

            que_res = None
            que_active_threads = None

            mngr.shutdown()
            mngr = None

        else:
            self._lock = Lock()

            for i in range(n_init_temps):

                search_attempts += 1

                acpt_rates_temps.append(
                    self._get_acpt_rate_and_temp(
                        (i, init_temps[i], None, None)))

                if (acpt_rates_temps[-1][0] >=
                    self._sett_ann_auto_init_temp_acpt_max_bd_hi):

                    break

            self._lock = None

        if self._vb:
            print_el()

        assert search_attempts == len(acpt_rates_temps)

        acpt_rates_temps = np.array(acpt_rates_temps)

        assert (
            np.any(acpt_rates_temps[:, 0] >=
                self._sett_ann_auto_init_temp_acpt_bd_lo) and
            np.any(acpt_rates_temps[:, 0] <=
                self._sett_ann_auto_init_temp_acpt_bd_hi)), (
            f'Could not find temperatures that give a suitable acceptance '
            f'rate.\n'
            f'Acceptance rates\' and temperatures\' matrix:\n'
            f'{acpt_rates_temps}')

        ann_init_temp, interp_acpt_rates_temps = (
            self._get_auto_init_temp_and_interped_temps(acpt_rates_temps))

        self._plot_acpt_rate_temps(
            interp_acpt_rates_temps,
            acpt_rates_temps,
            ann_init_temp)

        assert (
            self._sett_ann_auto_init_temp_temp_bd_lo <=
            ann_init_temp <=
            self._sett_ann_auto_init_temp_temp_bd_hi), (
                f'Interpolated initialization temperature '
                f'{ann_init_temp:6.2E} is out of bounds!')

        assert ann_init_temp > 0, (
            f'Interpolated initialization temperature {ann_init_temp:6.2E} '
            f'is negative!')

        assert (
            self._sett_ann_auto_init_temp_temp_bd_lo <=
            ann_init_temp <=
            self._sett_ann_auto_init_temp_temp_bd_hi), (
                f'Initialization temperature {ann_init_temp:5.3E} out of '
                f'bounds!')

        self._alg_ann_runn_auto_init_temp_search_flag = False

        self._sett_ann_init_temp = ann_init_temp

        end_tm = default_timer()

        if self._vb:
            print_sl()

            time_tem_str = show_formatted_elapsed_time(end_tm - beg_tm)

            print(
                f'Found initialization temperature of '
                f'{self._sett_ann_init_temp:5.3E} in {time_tem_str} '
                f'using {search_attempts} attempts.')

            print_el()
        return
