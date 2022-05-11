'''
Import and add calculations to raw data.
'''

from MemTempl_analysis_config import path_rawDat, exclude_outliers
from MemTempl_analysis_functions import import_all_data, exclude_outlier_data

path_data = path_rawDat

one_session = []
two_sessions = ['Subj_008', 'Subj_009', 'Subj_010', 'Subj_011', 'Subj_012', 'Subj_013', 'Subj_014', 'Subj_015', 'Subj_016', 'Subj_017', 'Subj_018',
                'Subj_019', 'Subj_020', 'Subj_021', 'Subj_022', 'Subj_023', 'Subj_024', 'Subj_025', 'Subj_026', 'Subj_027', 'Subj_029', 'Subj_030', 'Subj_031']
subjects = two_sessions + one_session
subjects_template_reproduction_order_unfixed = ['Subj_008', 'Subj_012', 'Subj_009', 'Subj_010', 'Subj_013', 'Subj_011']

qualCheck = False
binsize_cwccw = 6
binsfromto_cwccw = 45

data, angletraining_lengths = import_all_data(subjects, one_session, two_sessions, subjects_template_reproduction_order_unfixed, path_data, qualCheck=qualCheck)

data, templ_corr, memitem_corr, cwccw_outliers = exclude_outlier_data(
    data, binsize_cwccw, binsfromto_cwccw, exclude=exclude_outliers)

#%%
'''
Prepare data to export to R/JASP and for some plots, or recover already prepared files.
'''
import numpy as np
import pandas as pd
from os.path import exists
from pathlib import Path
from MemTempl_analysis_config import path_results, filename_list, split
from MemTempl_analysis_functions import remove_outliers_SD, prepare_statistics_MemItem_Repr_performance, prepare_statistics_Template_Repr_performance, prepare_statistics_Template_judgment, prepare_statistics_serDep

# fileexist = []
# for filei in range(len(filename_list)):
#     if exists(path_results/filename_list[filei]):
#         fileexist.append(1)
#     else:
#         fileexist.append(0)

# if 1 in fileexist:
#     for filei in range(len(filename_list)):
        
save = True

if not exists(path_results/"MemItem_Rep_performance.csv"):
    memitem_statistics_data, memitem_means, memitem_sds = prepare_statistics_MemItem_Repr_performance(data, save = save)
else:
    memitem_statistics_data = pd.read_csv(Path(path_results/"MemItem_Rep_performance.csv"))

if not exists(path_results/"Template_Rep_performance.csv"):
    template_statistics_data = prepare_statistics_Template_Repr_performance(data, save = save)
else:
    template_statistics_data = pd.read_csv(Path(path_results/"Template_Rep_performance.csv"))

if not exists(path_results/"Template_Judg_performance.csv"):
    judgment_statistics_data = prepare_statistics_Template_judgment(data, save = save)
else:
    judgment_statistics_data = pd.read_csv(Path(path_results/"Template_Judg_performance.csv"))

if not exists(path_results/"Modelfree_dependence_measure.csv"):
    split = split
    n_permutations = 5000
    
    data_clean, _ = remove_outliers_SD(data, cutoff=3, qualCheck=False)
    modelfree_dependence_statistics_data = prepare_statistics_serDep(data_clean, split = split, n_permutations = n_permutations, save = save)
else:
    modelfree_dependence_statistics_data = pd.read_csv(Path(path_results/"Modelfree_dependence_measure.csv"))

##NEXT UP: finish re-doing template reproduction, judgment, and SD R-code

#%%
'''
Plot Gabor reproduction performance
'''
from MemTempl_analysis_functions import plot_memitem_reproduction_error

split_passes = True
plot_single_subjects = False

plot_memitem_reproduction_error(data, split_passes, plot_single_subjects)

#%%
'''
Plot Template reproduction performance
'''
from MemTempl_analysis_functions import plot_template_reproduction_error

plot_single_subjects = False

plot_template_reproduction_error(data, plot_single_subjects = plot_single_subjects)

#%%
'''
Plot judgment task performance (after Stokes 2021) and differences over sessions/colors (Thesis figure 8)
'''
from MemTempl_analysis_functions import plot_active_latent_together, plot_judgment_results

plot_single_subjects = True
binsize = 6
fit_fromto = 40
response = "ccw"

plot_active_latent_together(data, plot_single_subjects = plot_single_subjects, response = response, binsize = binsize, fit_fromto = fit_fromto)

plot_judgment_results(judgment_statistics_data)

#%%
'''
Plot serial dependence curves with best-fitting DoG curves
'''
import numpy as np
from MemTempl_analysis_functions import remove_outliers_SD, plotMovingAverage
from MemTempl_analysis_functions_fits import return_dogparams, Derivative_of_Gaussian

data_clean, perc_removed = remove_outliers_SD(data, cutoff=3, qualCheck=False)

x = np.linspace(-90,90,181)
conditions = ['SD_classic','SD_resp','SD_actTempl','SD_latTempl']
pass_colors = ['k','pink']

passes = 1

single_subject_curves = {}
mean_curves = {}
best_DoG_parameters = {}

for condi in conditions:
    data_curve_ss, data_curve_mean,ax,xlab = plotMovingAverage(data=data_clean, bin_width=20, SD_type=condi, plot_at_all=True,
                                                       plot_single_subjects=False, return_single_subjects=False,
                                                       mean_type='cross', passes=passes, n_back=4, split=[90])
    
    single_subject_curves[condi] = data_curve_ss
    mean_curves[condi] = data_curve_mean
    
    best_DoG_parameters[condi] = return_dogparams(data_curve_mean, xrange = 90, rangelim = 90, plot = False, SD_type = '', fittingsteps = 50)

    for i in range(passes):
        amp = best_DoG_parameters[condi][i,0]
        width = best_DoG_parameters[condi][i,1]
        ax.plot(x, Derivative_of_Gaussian(x,amp,width), c=pass_colors[i], linewidth = 2.5)

#%%
'''
Plot model-free measures of dependence
'''
from MemTempl_analysis_config import split
from MemTempl_analysis_functions import plot_single_subject_modelfree_dependence, plot_mean_modelfree_dependence

#Plot single-subject measures of dependence
plot_mean_modelfree_dependence(modelfree_dependence_statistics_data, split= split, title = True, plot_singlesubject = False, savefig = False)

#Plot mean measures of dependence
sign_subjects = plot_single_subject_modelfree_dependence(modelfree_dependence_statistics_data, title=True)