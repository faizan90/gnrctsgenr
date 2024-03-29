'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from math import factorial
from timeit import default_timer
from itertools import combinations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from matplotlib.colors import Normalize

from fcopulas import fill_bi_var_cop_dens, get_asymms_nd_v2_raw_cy

from .aa_setts import get_mpl_prms, set_mpl_prms

from ...misc import get_lagged_pair_corrs_dict, get_lagged_pair_asymms_dict

plt.ioff()


class GTGPlotMultiSite:

    '''
    Supporting class of Plot.

    Multi-site plots.
    '''

    def __init__(self):

        return

    def _plot_asymms_nd(self):

        '''
        Plot N-Dimensional asymmetries.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_grp = h5_hdl[f'data_ref_rltzn']

        ref_probs = ref_grp['probs'][...].copy(order='c')

        ref_asymms = get_asymms_nd_v2_raw_cy(ref_probs)

        sim_asymmss = {}
        for rltzn_lab in sim_grp_main:

            sim_asymms = get_asymms_nd_v2_raw_cy(
                sim_grp_main[f'{rltzn_lab}/probs'][...].copy(order='c'))

            sim_asymmss[rltzn_lab] = sim_asymms

        x_crds = np.arange(ref_probs.shape[1])

        plt.figure()
        for asymm_idx in range(ref_asymms.shape[0]):

            plt.plot(
                x_crds,
                ref_asymms[asymm_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                plt.plot(
                    x_crds,
                    sim_asymmss[rltzn_lab][asymm_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Component value (-)')

            plt.xlabel('Component index (-)')

            plt.xticks(x_crds, x_crds)

            out_name = f'ms__nd_asymms_{asymm_idx}.png'

            plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

            plt.clf()

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site nD asymmetries '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_gnrc_pair_corrs(self, corr_type, lags):

        assert corr_type in ('pearson', 'spearman'), corr_type

        assert len(lags)

        assert all([isinstance(x, int) for x in lags])

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        ref_grp = h5_hdl[f'data_ref_rltzn']

        ref_pair_corrs = get_lagged_pair_corrs_dict(
            ref_grp['data'][...], corr_type, lags)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        sim_pair_corrs_all = []
        for rltzn_lab in sim_grp_main:
            sim_pair_corrs = get_lagged_pair_corrs_dict(
                sim_grp_main[f'{rltzn_lab}/data'][...], corr_type, lags)

            sim_pair_corrs_all.append(sim_pair_corrs)

        h5_hdl.close()

        scatt_min, scatt_max = np.inf, -np.inf

        plt.figure()
        for i, lag in enumerate(lags):

            scatt_min = min([min(ref_pair_corrs[lag]), scatt_min])
            scatt_max = max([max(ref_pair_corrs[lag]), scatt_max])

            leg_flag = True
            for j in range(len(sim_pair_corrs_all)):

                if leg_flag:
                    label = f'lag: {lag:+d}'

                else:
                    label = None

                plt.scatter(
                    ref_pair_corrs[lag],
                    sim_pair_corrs_all[j][lag],
                    alpha=plt_sett.alpha_1,
                    c=f'C{i}',
                    edgecolors='none',
                    label=label)

                leg_flag = False

                scatt_min = min([min(sim_pair_corrs_all[j][lag]), scatt_min])
                scatt_max = max([max(sim_pair_corrs_all[j][lag]), scatt_max])

        plt.xlabel('Reference')
        plt.ylabel('Simulated')

        scatt_min -= 0.05
        scatt_max += 0.05

        plt.plot(
            [scatt_min, scatt_max],
            [scatt_min, scatt_max],
            alpha=plt_sett.alpha_1,
            ls='--',
            c='k')

        plt.xlim(scatt_min, scatt_max)
        plt.ylim(scatt_min, scatt_max)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.gca().set_aspect('equal')

        out_name = f'ms__cross_corrs_scatter_{corr_type}.png'

        plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

        plt.close('all')

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site pair {corr_type} correlations'
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_gnrc_pair_asymms(self, asymm_type, lags):

        assert asymm_type in ('order', 'directional'), asymm_type

        assert len(lags)

        assert all([isinstance(x, int) for x in lags])

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        ref_grp = h5_hdl[f'data_ref_rltzn']

        ref_pair_asymms = get_lagged_pair_asymms_dict(
            ref_grp['data'][...], asymm_type, lags)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        sim_pair_asymms_all = []
        for rltzn_lab in sim_grp_main:
            sim_pair_asymms = get_lagged_pair_asymms_dict(
                sim_grp_main[f'{rltzn_lab}/data'][...], asymm_type, lags)

            sim_pair_asymms_all.append(sim_pair_asymms)

        h5_hdl.close()

        scatt_min, scatt_max = np.inf, -np.inf

        plt.figure()
        for i, lag in enumerate(lags):

            scatt_min = min([min(ref_pair_asymms[lag]), scatt_min])
            scatt_max = max([max(ref_pair_asymms[lag]), scatt_max])

            leg_flag = True
            for j in range(len(sim_pair_asymms_all)):

                if leg_flag:
                    label = f'lag: {lag:+d}'

                else:
                    label = None

                plt.scatter(
                    ref_pair_asymms[lag],
                    sim_pair_asymms_all[j][lag],
                    alpha=plt_sett.alpha_1,
                    c=f'C{i}',
                    edgecolors='none',
                    label=label)

                leg_flag = False

                scatt_min = min([min(sim_pair_asymms_all[j][lag]), scatt_min])
                scatt_max = max([max(sim_pair_asymms_all[j][lag]), scatt_max])

        plt.xlabel('Reference')
        plt.ylabel('Simulated')

        scatt_min -= 0.05
        scatt_max += 0.05

        plt.plot(
            [scatt_min, scatt_max],
            [scatt_min, scatt_max],
            alpha=plt_sett.alpha_1,
            ls='--',
            c='k')

        plt.xlim(scatt_min, scatt_max)
        plt.ylim(scatt_min, scatt_max)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.gca().set_aspect('equal')

        out_name = f'ms__cross_asymm_scatter_{asymm_type}.png'

        plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

        plt.close('all')

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site pair {asymm_type} asymmetries'
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_gnrc_ms_cumm_ft(self, var_label):

        '''
        Maximum FT cross correlation.
        '''

        assert var_label in ('data', 'probs'), var_label

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_grp = h5_hdl[f'data_ref_rltzn']

        ref_var_ft = ref_grp[f'{var_label}_ms_ft'][...]

        ref_periods = (ref_var_ft.size * 2) / (
            np.arange(1, ref_var_ft.size + 1))

        plt.figure()

        plt.semilogx(
            ref_periods,
            ref_var_ft,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        sim_periods = None
        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_var_ft = sim_grp_main[
                f'{rltzn_lab}/{var_label}_ms_ft'][...]

            if sim_periods is None:
                sim_periods = (sim_var_ft.size * 2) / (
                    np.arange(1, sim_var_ft.size + 1))

            plt.semilogx(
                sim_periods,
                sim_var_ft,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.legend(framealpha=0.7)

        plt.ylabel(f'Cummulative {var_label} FT correlation')

        plt.xlabel(f'Period (steps)')

        plt.xlim(plt.xlim()[::-1])

        out_name = f'ms__ms_ft_{var_label}.png'

        plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multisite-site {var_label} FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_nD_vars(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_1D_vars_wider

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_grp = h5_hdl[f'data_ref_rltzn']

        axes = plt.subplots(2, 3, squeeze=False, sharex=True, sharey=False)[1]

        axes[0, 0].scatter(
            0,
            ref_grp['scorrs_ms'][0],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            edgecolors='none',
            lw=plt_sett.lw_2,
            label='ref')

        axes[0, 1].scatter(
            0,
            ref_grp['ecop_etpy_ms'][0],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            edgecolors='none',
            lw=plt_sett.lw_2,
            label='ref')

        axes[1, 2].scatter(
            0,
            ref_grp['scorrs_ms'][1],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            edgecolors='none',
            lw=plt_sett.lw_2,
            label='ref')

        min_scorr = min(ref_grp['scorrs_ms'][0], ref_grp['scorrs_ms'][1])
        max_scorr = max(ref_grp['scorrs_ms'][0], ref_grp['scorrs_ms'][1])

        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_grp = sim_grp_main[f'{rltzn_lab}']

            axes[0, 0].scatter(
                1,
                sim_grp['scorrs_ms'][0],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                edgecolors='none',
                lw=plt_sett.lw_1,
                label=label)

            axes[0, 1].scatter(
                1,
                sim_grp['ecop_etpy_ms'][0],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                edgecolors='none',
                lw=plt_sett.lw_1,
                label=label)

            axes[1, 2].scatter(
                1,
                sim_grp['scorrs_ms'][1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                edgecolors='none',
                lw=plt_sett.lw_1,
                label=label)

            min_scorr = min(
                min_scorr , sim_grp['scorrs_ms'][0], sim_grp['scorrs_ms'][1])

            max_scorr = max(
                max_scorr , sim_grp['scorrs_ms'][0], sim_grp['scorrs_ms'][1])

            leg_flag = False

        axes[0, 0].axhline(
            ref_grp['scorrs_ms'][0],
            ls='--',
            alpha=plt_sett.alpha_1,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_1)

        axes[0, 1].axhline(
            ref_grp['ecop_etpy_ms'][0],
            ls='--',
            alpha=plt_sett.alpha_1,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_1)

        axes[1, 2].axhline(
            ref_grp['scorrs_ms'][1],
            ls='--',
            alpha=plt_sett.alpha_1,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_1)

        # Having all of the x-axes shared, we just adjust a single one.
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_xticklabels(['ref', 'sim'])
        axes[0, 0].set_xlim([-0.5, +1.5])

        if min_scorr >= 0:
            min_scorr = -0.05

        else:
            min_scorr = -1.05

        if max_scorr >= 0:
            max_scorr = 1.05

        else:
            max_scorr = 0.05

        axes[0, 0].set_ylim(min_scorr, max_scorr)
        axes[1, 2].set_ylim(min_scorr, max_scorr)
        axes[0, 1].set_ylim(-0.05, 1.05)

        axes[0, 0].grid()
        # axes[1, 0].grid()
        # axes[1, 1].grid()
        axes[0, 1].grid()
        # axes[0, 2].grid()
        axes[1, 2].grid()

        axes[0, 0].set_axisbelow(True)
        axes[1, 0].set_axisbelow(True)
        axes[1, 1].set_axisbelow(True)
        axes[0, 1].set_axisbelow(True)
        axes[0, 2].set_axisbelow(True)
        axes[1, 2].set_axisbelow(True)

        # axes[0, 0].set_axis_off()
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()
        # axes[0, 1].set_axis_off()
        axes[0, 2].set_axis_off()
        # axes[1, 2].set_axis_off()

        axes[0, 0].legend(framealpha=0.7)
#             axes[1, 0].legend(framealpha=0.7)
#             axes[1, 1].legend(framealpha=0.7)
#             axes[0, 1].legend(framealpha=0.7)
#             axes[0, 2].legend(framealpha=0.7)
#             axes[1, 2].legend(framealpha=0.7)

        axes[0, 0].set_ylabel('Spearman correlation (+ve)')

        # axes[1, 0].set_ylabel('Asymmetry (Type - 1)')
        #
        # axes[1, 1].set_ylabel('Asymmetry (Type - 2)')
        #
        axes[0, 1].set_ylabel('Entropy')
        #
        # axes[0, 2].set_ylabel('Pearson correlation')

        axes[1, 2].set_ylabel('Spearman correlation (-ve)')

        plt.tight_layout()

        fig_name = f'ms__summary.png'

        plt.savefig(str(self._ms_dir / fig_name), bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site nD lumped statistics '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_gnrc_cross_cmpos_ft(self, var_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ms__cmpos_ft_cumsum_{var_label}'

        comb_size = 2
        n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

        n_combs = int(
            factorial(n_data_labels) /
            (factorial(comb_size) *
             factorial(n_data_labels - comb_size)))

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_vals = h5_hdl[
            f'data_ref_rltzn/mult_{var_label}_cmpos_ft_dict'][:]

        ref_periods = ((ref_vals.size - n_combs) * 2) / (
            np.arange(1, ref_vals.size - n_combs + 1))

        add_ref_periods = []
        for i in range(n_combs):
            add_ref_periods.append(ref_periods[0] * (2 + i))

        ref_periods = np.concatenate((add_ref_periods[::-1], ref_periods))

        assert ref_vals.size == ref_periods.size

        # cumm ft corrs, sim_ref
        plt.figure()

        plt.semilogx(
            ref_periods,
            ref_vals,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        plt.axvline(
            ref_periods[n_combs],
            0,
            1,
            color='b',
            alpha=plt_sett.alpha_2,
            lw=plt_sett.lw_2)

        sim_periods = None
        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_vals = sim_grp_main[
                f'{rltzn_lab}/mult_{var_label}_cmpos_ft'][:]

            if sim_periods is None:
                sim_periods = ((sim_vals.size - n_combs) * 2) / (
                    np.arange(1, sim_vals.size - n_combs + 1))

                add_sim_periods = []
                for i in range(n_combs):
                    add_sim_periods.append(sim_periods[0] * (2 + i))

                sim_periods = np.concatenate(
                    (add_sim_periods[::-1], sim_periods))

            plt.semilogx(
                sim_periods,
                sim_vals,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.legend(framealpha=0.7)

        plt.xlim(plt.xlim()[::-1])

        plt.ylim(-1, +2)

        plt.ylabel('Cummulative cmpos FT correlation')
        plt.xlabel('Period (steps)')

        out_name = f'{out_name_pref}.png'

        plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site {var_label} cmpos FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_gnrc_cdfs(self, var_label, x_ax_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ms__cross_{var_label}_cdfs'

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        combs = combinations(data_labels, 2)

        loop_prod = combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for cols in loop_prod:

            assert len(cols) == 2

            ref_probs = h5_hdl[
                f'data_ref_rltzn/{var_label}_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_y'][:]

            ref_vals = h5_hdl[
                f'data_ref_rltzn/{var_label}_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_x'][:]

            plt.figure()

            plt.plot(
                ref_vals,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/{var_label}_{cols[0]}_{cols[1]}'][:]

                sim_probs = rankdata(sim_vals) / (sim_vals.size + 1)

                plt.plot(
                    sim_vals,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')
            plt.xlabel(f'{x_ax_label}')

            out_name = f'{out_name_pref}_{"_".join(cols)}.png'

            plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site {var_label} CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_denss_cntmnt(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_denss_cntmnt

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_ecop_dens_bins = h5_hdl['settings'].attrs['sett_obj_ecop_dens_bins']

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_beta = plt.get_cmap('Accent').resampled(3)  # plt.get_cmap(plt.rcParams['image.cmap'])

        cmap_beta.colors[1,:] = [1, 1, 1, 1]

        ref_ecop_dens_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            np.nan,
            dtype=np.float64)

        tem_ecop_dens_arr = np.empty_like(ref_ecop_dens_arr)

        cntmnt_ecop_dens_arr = np.empty_like(ref_ecop_dens_arr)

        cmap_mappable_beta = plt.cm.ScalarMappable(
            norm=Normalize(-1, +1, clip=True), cmap=cmap_beta)

        cmap_mappable_beta.set_array([])

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/probs'][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/probs'][:, di_b]

            fill_bi_var_cop_dens(probs_a, probs_b, ref_ecop_dens_arr)

            sim_ecop_dens_mins_arr = np.full(
                (n_ecop_dens_bins, n_ecop_dens_bins),
                +np.inf,
                dtype=np.float64)

            sim_ecop_dens_maxs_arr = np.full(
                (n_ecop_dens_bins, n_ecop_dens_bins),
                -np.inf,
                dtype=np.float64)

            for rltzn_lab in sim_grp_main:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'][:, di_b]

                fill_bi_var_cop_dens(probs_a, probs_b, tem_ecop_dens_arr)

                sim_ecop_dens_mins_arr = np.minimum(
                    sim_ecop_dens_mins_arr, tem_ecop_dens_arr)

                sim_ecop_dens_maxs_arr = np.maximum(
                    sim_ecop_dens_maxs_arr, tem_ecop_dens_arr)

            cntmnt_ecop_dens_arr[:] = 0.0

            cntmnt_ecop_dens_arr[
                ref_ecop_dens_arr < sim_ecop_dens_mins_arr] = -1

            cntmnt_ecop_dens_arr[
                ref_ecop_dens_arr > sim_ecop_dens_maxs_arr] = +1

            fig_suff = f'ms__cross_ecop_dens_cnmnt_{dl_a}_{dl_b}'

            fig, axes = plt.subplots(1, 1, squeeze=False)

            row, col = 0, 0

            dx = 1.0 / (cntmnt_ecop_dens_arr.shape[1] + 1.0)
            dy = 1.0 / (cntmnt_ecop_dens_arr.shape[0] + 1.0)

            y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

            axes[row, col].pcolormesh(
                x,
                y,
                cntmnt_ecop_dens_arr,
                vmin=-1,
                vmax=+1,
                alpha=plt_sett.alpha_1,
                cmap=cmap_beta,
                shading='auto')

            axes[row, col].set_aspect('equal')

            axes[row, col].set_ylabel('Probability')
            axes[row, col].set_xlabel('Probability')

            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)

            cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

            cb = plt.colorbar(
                mappable=cmap_mappable_beta,
                cax=cbaxes,
                orientation='horizontal',
                label='Empirical copula density containment',
                alpha=plt_sett.alpha_1,
                ticks=[-1, 0, +1],
                drawedges=False)

            cb.ax.set_xticklabels(['Too hi.', 'Within', 'Too lo.'])

            plt.savefig(
                str(self._ms_dir /
                    f'ms__cross_ecops_denss_cmpr_{fig_suff}.png'),
                bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop density containment '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_denss(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_denss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_ecop_dens_bins = h5_hdl['settings'].attrs['sett_obj_ecop_dens_bins']

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        ecop_dens_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            np.nan,
            dtype=np.float64)

        best_rltzn_labs = self._get_best_obj_vals_srtd_sim_labs(sim_grp_main)

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/probs'
                ][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/probs'
                ][:, di_b]

            fill_bi_var_cop_dens(probs_a, probs_b, ecop_dens_arr)

            fig_suff = f'ref_{dl_a}_{dl_b}'

            vmin = 0.0
#             vmax = np.mean(ecop_dens_arr) * 2.0
            vmax = np.max(ecop_dens_arr) * 0.85

            cmap_mappable_beta = plt.cm.ScalarMappable(
                norm=Normalize(vmin / 100, vmax / 100, clip=True),
                cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            args = (
                fig_suff,
                vmin,
                vmax,
                ecop_dens_arr,
                cmap_mappable_beta,
                self._ms_dir,
                plt_sett)

            self._plot_cross_ecop_denss_base(args)

            plot_ctr = 0
            for rltzn_lab in best_rltzn_labs:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'
                    ][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'
                    ][:, di_b]

                fill_bi_var_cop_dens(probs_a, probs_b, ecop_dens_arr)

                fig_suff = f'sim_{dl_a}_{dl_b}_{rltzn_lab}'

                args = (
                    fig_suff,
                    vmin,
                    vmax,
                    ecop_dens_arr,
                    cmap_mappable_beta,
                    self._ms_dir,
                    plt_sett)

                self._plot_cross_ecop_denss_base(args)

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop densities '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cross_ecop_denss_base(args):

        (fig_suff,
         vmin,
         vmax,
         ecop_dens_arr,
         cmap_mappable_beta,
         out_dir,
         plt_sett) = args

        fig, axes = plt.subplots(1, 1, squeeze=False)

        row, col = 0, 0

        dx = 1.0 / (ecop_dens_arr.shape[1] + 1.0)
        dy = 1.0 / (ecop_dens_arr.shape[0] + 1.0)

        y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

        axes[row, col].pcolormesh(
            x,
            y,
            ecop_dens_arr,
            vmin=vmin,
            vmax=vmax,
            alpha=plt_sett.alpha_1,
            shading='auto')

        axes[row, col].set_aspect('equal')

        axes[row, col].set_ylabel('Probability')
        axes[row, col].set_xlabel('Probability')

        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Empirical copula density',
            extend='max',
            alpha=plt_sett.alpha_1,
            drawedges=False)

        plt.savefig(
            str(out_dir / f'ms__cross_ecops_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cross_ft_corrs(self, data_type):

        '''
        Meant for pairs right now.
        '''

        assert data_type in ('data', 'probs'), data_type

        if data_type == 'data':
            data_type_label = 'data_ms_pair_ft'

        elif data_type == 'probs':
            data_type_label = 'probs_ms_pair_ft'

        else:
            raise NotImplementedError

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        data_label_idx_combs = combinations(data_labels, 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        comb_ctr = 0
        for dl_a, dl_b in loop_prod:

            ref_ft_cumm_corr = h5_hdl[
                f'data_ref_rltzn/{data_type_label}'][:, comb_ctr]

            ref_periods = (ref_ft_cumm_corr.size * 2) / (
                np.arange(1, ref_ft_cumm_corr.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_ft_cumm_corr,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_ft_cumm_corr = sim_grp_main[
                    f'{rltzn_lab}/{data_type_label}'][:, comb_ctr]

                if sim_periods is None:
                    sim_periods = (sim_ft_cumm_corr.size * 2) / (
                        np.arange(1, sim_ft_cumm_corr.size + 1))

                plt.semilogx(
                    sim_periods,
                    sim_ft_cumm_corr,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative correlation')

            plt.xlabel(f'Period (steps)')

            plt.xlim(plt.xlim()[::-1])

            plt.ylim(-1, +1)

            out_name = f'ms__ft_cross_cumm_corrs_{data_type}_{dl_a}_{dl_b}.png'

            plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

            plt.close()

            comb_ctr += 1

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross pair {data_type} FT correlations '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_scatter(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_str = 'jet'

        cmap_beta = plt.get_cmap(cmap_str)

        best_rltzn_labs = self._get_best_obj_vals_srtd_sim_labs(sim_grp_main)

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/probs'][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/probs'][:, di_b]

            fig_suff = f'ref_{dl_a}_{dl_b}'

            cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            ref_timing_ser = np.arange(
                1.0, probs_a.size + 1.0) / (probs_a.size + 1.0)

            ref_clrs = plt.get_cmap(cmap_str)(ref_timing_ser)

            sim_timing_ser = ref_timing_ser
            sim_clrs = ref_clrs

            args = (
                probs_a,
                probs_b,
                fig_suff,
                self._ms_dir,
                plt_sett,
                cmap_mappable_beta,
                ref_clrs)

            self._plot_cross_ecop_scatter_base(args)

            plot_ctr = 0
            for rltzn_lab in best_rltzn_labs:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'][:, di_b]

                if ref_timing_ser.size != sim_clrs.shape[0]:
                    sim_timing_ser = np.arange(
                        1.0, probs_a.size + 1.0) / (probs_a.size + 1.0)

                    sim_clrs = plt.get_cmap(cmap_str)(sim_timing_ser)

                fig_suff = f'sim_{dl_a}_{dl_b}_{rltzn_lab}'

                args = (
                    probs_a,
                    probs_b,
                    fig_suff,
                    self._ms_dir,
                    plt_sett,
                    cmap_mappable_beta,
                    sim_clrs)

                self._plot_cross_ecop_scatter_base(args)

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop scatters '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cross_ecop_scatter_base(args):

        (probs_a,
         probs_b,
         fig_suff,
         out_dir,
         plt_sett,
         cmap_mappable_beta,
         clrs) = args

        fig, axes = plt.subplots(1, 1, squeeze=False)

        row, col = 0, 0

        axes[row, col].scatter(
            probs_a,
            probs_b,
            c=clrs,
            edgecolors='none',
            alpha=plt_sett.alpha_1)

        axes[row, col].grid()

        axes[row, col].set_axisbelow(True)

        axes[row, col].set_aspect('equal')

        axes[row, col].set_ylabel('Probability')
        axes[row, col].set_xlabel('Probability')

        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Timing',
            drawedges=False)

        plt.savefig(
            str(out_dir / f'ms__cross_ecops_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return
