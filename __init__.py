'''
Created on Dec 27, 2019

@author: Faizan
'''
import os
from multiprocessing import current_process

# Due to shitty tkinter errors.
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from .core import (
    GTGBase,
    GTGData,
    GTGSettings,
    GTGPrepareRltznRef,
    GTGPrepareRltznSim,
    GTGPrepareBase,
    GTGPrepareTfms,
    GTGPrepareCDFS,
    GTGPrepareUpdate,
    GTGPrepare,
    GTGAlgBase,
    GTGAlgObjective,
    GTGAlgIO,
    GTGAlgLagNthWts,
    GTGAlgLabelWts,
    GTGAlgAutoObjWts,
    GTGAlgRealization,
    GTGAlgTemperature,
    GTGAlgMisc,
    GTGAlgorithm,
    GTGSave,
    )

from .core import (
    get_mpl_prms,
    set_mpl_prms,
    GTGPlotBase,
    GTGPlotOSV,
    GTGPlotSingleSite,
    GTGPlotMultiSite,
    GTGPlotSingleSiteQQ,
    GenericTimeSeriesGeneratorPlot,
    )

from .misc import roll_real_2arrs, show_formatted_elapsed_time, ret_mp_idxs

current_process().authkey = 'gnrctsgenr'.encode(
    encoding='utf_8', errors='strict')
