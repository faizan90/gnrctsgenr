'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''
import psutil
import numpy as np
from scipy.stats import rankdata

from fcopulas import (
    fill_bin_idxs_ts,
    fill_bin_dens_1d,
    fill_bin_dens_2d,
    fill_etpy_lcl_ts)

PRINT_LINE_STR = 79 * '#'

SCI_N_ROUND = 4


def sci_round(data):

    assert data.ndim == 1

    round_data = np.array(
        [np.format_float_scientific(data[i], precision=SCI_N_ROUND)
         for i in range(data.size)], dtype=float)

    assert np.all(np.isfinite(round_data))
    assert np.all(round_data >= 0)

    return round_data


def print_sl():

    print(2 * '\n', PRINT_LINE_STR, sep='')
    return


def print_el():

    print(PRINT_LINE_STR)
    return


def get_n_cpus():

    phy_cores = psutil.cpu_count(logical=False)
    log_cores = psutil.cpu_count()

    if phy_cores < log_cores:
        n_cpus = phy_cores

    else:
        n_cpus = log_cores - 1

    n_cpus = max(n_cpus, 1)

    return n_cpus


def ret_mp_idxs(n_vals, n_cpus):

    assert n_vals > 0

    idxs = np.linspace(
        0, n_vals, min(n_vals + 1, n_cpus + 1), endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def roll_real_2arrs(arr1, arr2, lag, rerank_flag=False):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[-lag:].copy()
        arr2 = arr2[:+lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2


def get_local_entropy_ts_cy(probs_x, probs_y, n_bins):

    bins_ts_x = np.empty_like(probs_x, dtype=np.uint32)
    bins_ts_y = np.empty_like(probs_y, dtype=np.uint32)

    bins_dens_x = np.empty(n_bins, dtype=float)
    bins_dens_y = np.empty(n_bins, dtype=float)

    bins_dens_xy = np.empty((n_bins, n_bins), dtype=float)

    lcl_etpy_ts = np.empty_like(probs_x, dtype=float)

    fill_bin_idxs_ts(probs_x, bins_ts_x, n_bins)
    fill_bin_idxs_ts(probs_y, bins_ts_y, n_bins)

    fill_bin_dens_1d(bins_ts_x, bins_dens_x)
    fill_bin_dens_1d(bins_ts_y, bins_dens_y)

    fill_bin_dens_2d(bins_ts_x, bins_ts_y, bins_dens_xy)

    fill_etpy_lcl_ts(
        bins_ts_x,
        bins_ts_y,
        bins_dens_x,
        bins_dens_y,
        lcl_etpy_ts,
        bins_dens_xy)

    return lcl_etpy_ts