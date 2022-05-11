def import_all_data(subject_list, one_session, two_sessions, subjects_template_reproduction_order_unfixed, path_data, qualCheck = False):
    import numpy as np
    import pandas as pd
    from MemTempl_analysis_functions_basic import import_data, fix_template_reproduction_order, remove_half_pass, add_memitem_reproduction_error, add_template_reproduction_error_column, distance_cued_template_memitem, add_memitem_previousrep_angle,add_corrected_resp_error, getAngularDistance, determine_switch_nonswitch, cwccw_performance_column,add_distortion_condition, get_memitem_RT, find_trial_length
    
    """
    Function to import raw data files and perform initial analysis steps/calculations.
    
    Returns merged dataframe of all subjects.
    
    :param subject_list: List of codes for all subjects to be evaluated.
    :param one_session: List of codes for all subjects that performed only one session.
    :param two_sessions: List of codes for all subjects that performed both sessions.
    :param subjects_template_reproduction_order_unfixed: List of codes for all subjects 
                                                        that need special data cleaning 
                                                        due to procedural error.
    :param path_data: Path to raw data files.
    :param qualCheck: Whether to check if data cleaning step in add_corrected_resp_error() worked.
    
    """
    data_list = []
    angletraining_lengths = {}
    
    for i in range(len(subject_list)):
        if subject_list[i] == 'Subj_014':
            df_one = import_data(path_data, subject_list[i],doublepass = 0)
            df_one = fix_template_reproduction_order(df_one)
            df_one['pass_no'] = np.ones((len(df_one)))
            file_name = path_data + "rawDat_Subj_014_Sess_002_final.csv"
            df_two = pd.read_csv(file_name, delimiter = ',')
            df_two['pass_no'] = np.repeat(2,len(df_two))
            
            df = pd.concat((df_one,df_two)).reset_index(drop=True)
        
        elif subject_list[i] == 'Subj_021':
            df = import_data(path_data, subject_list[i], doublepass = 1, other_file ="rawDat_Subj_021_Sess_002_aborted.csv" )
            df = remove_half_pass(df)
        
        elif subject_list[i] in two_sessions:
            df = import_data(path_data, subject_list[i],doublepass = 1)
            
        elif subject_list[i] in one_session:
            df = import_data(path_data, subject_list[i],doublepass = 0)
            
        if subject_list[i] in subjects_template_reproduction_order_unfixed:
            df = fix_template_reproduction_order(df)
    
        ##add all columns necessary for further analysis
        #subjectname column
        df.insert(0, 'Subject', subject_list[i])
        #memitem reproduction error column
        df, _, _ = add_memitem_reproduction_error(df)
        #template reproduction error columns
        df = add_template_reproduction_error_column(df)
        #memitem-template distance columns
        distance_active_item = distance_cued_template_memitem(df,'cwccw','active')
        df['distance_active_item'] = distance_active_item
        distance_latent_item = distance_cued_template_memitem(df,'cwccw','latent')
        df['distance_latent_item'] = distance_latent_item
        #memitem-previous trial's response distance column
        df['distance_previous_resp'] = add_memitem_previousrep_angle(df)
        #add column subtracting current-orientation-dependent response error from memitem response error (for use when analysing previous response influence on current response)
        df['Corr_MemItem_Resp_Error'] = add_corrected_resp_error(df, qualCheck = qualCheck)
        #add column denoting the distance between current memitem and n_back memitem
        
        #add column denoting the distance between templates
        df["Templ_distance"] = getAngularDistance(df["MemTempl1_angle"],df["MemTempl2_angle"])
        
        #add column denoting switch and non-switch trials (switch = 1)
        df['switch_trial'] = determine_switch_nonswitch(df)
        
        #memitem-previous template distance columns
        distance_prevact_item = distance_cued_template_memitem(df,'cwccw','active', prevorcurr = 'prev')
        df['distance_prevactive_item'] = distance_prevact_item
        distance_prevlat_item = distance_cued_template_memitem(df,'cwccw','latent', prevorcurr = 'prev')
        df['distance_prevlatent_item'] = distance_prevlat_item
        
        #cwccw performance column
        df["performance_cwccw"] = cwccw_performance_column(df)
        
        #add condition marker - 1 for expected distorted memitem (because of far distance), 0 for no expected distortion (because of close distance)
        df["prev_trial_distorted"] = add_distortion_condition(df)

        #add column with real memitem RT
        df["clean_MemItem_RT"] = get_memitem_RT(df)
        #add column with total trial length
        df["trial_length"] = find_trial_length(df)

        data_list.append(df)
        
        df_angle = import_data(path_data, subject_list[i], angletraining = True)
        angletraining_lengths[subject_list[i]] = len(df_angle)
        
    data_df = pd.concat(data_list).reset_index(drop=True)
    
    return data_df, angletraining_lengths

def exclude_outlier_data(data, exclude):
    import numpy as np
    from MemTempl_analysis_functions_basic import determine_template_reproduction_correlation, determine_memitem_reproduction_correlation, determine_cwccw_perccorrect
    
    """
    Function to perform inital analyses on data to determine and/or exclude outlier subjects.
    
    Returns (un-)cleaned dataframe and variables containing outlier measures.
    
    :param data: Full dataframe with all subjects' data.
    :param exclude: Whether to remove outlier subject data.
    
    """
    subs = np.unique(data.Subject)
    
    #Run functions determining outlier measures.
    templ_corr = determine_template_reproduction_correlation(data)
    memitem_corr = determine_memitem_reproduction_correlation(data)
    cwccw_outliers = determine_cwccw_perccorrect(data)
    
    #Variable to collect outlier subject codes.
    outliers = []
    
    #Check outlier measures and collect outliers.
    for subi, sub in enumerate(subs):
        if any(x < 0.5 for x in templ_corr[sub]) or memitem_corr[sub] < 0.5 or (cwccw_outliers[sub] < cwccw_outliers['mean']-(3*cwccw_outliers['sd']) or cwccw_outliers[sub] > cwccw_outliers['mean']+(3*cwccw_outliers['sd'])):#cwccw_outliers[sub][0] == 'True':
            outliers.append(sub)
    
    #Delete trials of subjects that are deemed outliers.
    if exclude:
        data = data[~data['Subject'].isin(outliers)]
    
    return data, templ_corr, memitem_corr, cwccw_outliers

def plot_memitem_reproduction_error(data, split_passes, plot_single_subjects):
    import pycircstat as circ
    import scipy.stats as stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    Function to plot performance measures of Gabor reproduction.
    
    :param data: Full dataframe with all subjects' data.
    :param split_passes: Whether to plot both sessions separately or pooled.
    :param plot_single_subjects: Whether to plot separate plots for each subject.
    
    """
    rng = np.random.default_rng()
    
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    bins = np.linspace(-90, 90, 37)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    
    colors_passes = ['b','r']
    colors_fillbetweens = ['cyan','orange']
    legends_passes = ['Session 1','Session 2']

    if split_passes:
        selTrials = [data[data['pass_no']==1],data[data['pass_no']==2]]
    else:
        selTrials = [data]
        
    n_cond = len(selTrials)
    
    #to collect single-subject performance histograms
    data_tmp = np.zeros((n_subs, n_cond, len(bins)-1))
    
    #iterate through subjects, determine response error histograms for each (and each pass) and save it
    for subi,sub in enumerate(subs):
        if plot_single_subjects:
            gif = plt.figure()
            axo = gif.add_subplot()
            sisu_title = str(sub)
        for condi in range(len(selTrials)):
            sel = selTrials[condi]
            tmp = sel.loc[sel['Subject'] == sub, 'MemItem_Repr_Error']
            if len(tmp) > 0 :
                tmp, _ = np.histogram(tmp, bins=bins, density=False) #compute histogram
                data_tmp[subi, condi, :] = tmp / np.sum(tmp)
                bin_centers = 0.5*(bins[1:]+bins[:-1])
                
                if plot_single_subjects:
                    axo.plot(bin_centers,data_tmp[subi, condi, :],linestyle = '-',color = colors_passes[condi])
                    axo.set_title(sisu_title)
    
    #prepare variable for mean and sem of all single-subject histograms (and one for each pass)                                    
    mean_tmp = np.zeros((len(selTrials),len(bins)-1))
    sem_tmp = np.zeros((len(selTrials), len(bins)-1))

    #determine mean and sem of all single-subject histograms for both passes
    tmp_mean = np.nanmean(data_tmp, axis=0)
    tmp_sem = stats.sem(data_tmp, axis = 0, nan_policy = 'omit')
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    #for each subject and pass, plot histogram in light color, and plot mean+sem in strong color
    for condi in range(len(selTrials)):
        for subi in range(n_subs):
            bin_centers = 0.5*(bins[1:]+bins[:-1])
            ax.plot(bin_centers,data_tmp[subi,condi,:],linestyle = '-',color=colors_passes[condi],alpha=0.1)
        mean_tmp[condi,:] = tmp_mean[condi]
        sem_tmp[condi,:] = tmp_sem[condi]
        ax.plot(bin_centers,mean_tmp[condi,:],linestyle = '-',color=colors_passes[condi],label=legends_passes[condi])
        plt.fill_between(bin_centers, mean_tmp[condi,:]-sem_tmp[condi,:],mean_tmp[condi,:]+sem_tmp[condi,:], alpha=1, edgecolor=colors_passes[condi], facecolor=colors_fillbetweens[condi])

        #format plot
        ax.set_xlabel('Gabor reproduction error [°]', fontsize=13, fontname='Arial')
        ax.set_xticks([-90,-60,-30,0,30,60,90])
        ax.set_xlim([-90,90])
        ax.set_ylabel('Probability', fontsize=13, fontname='Arial')
        ax.set_yticks([0,0.4])
        ax.set_ylim([0,0.4])
        ax.yaxis.set_label_coords(-0.02,.5)
        ax.vlines(0, 0, .4, colors='dimgray', linestyles='dotted')
        ax.spines['left'].set_color('k')
        ax.spines['bottom'].set_color('k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc=3, prop={'size': 10}, frameon = False, bbox_to_anchor=(0, 0.85))
        fig.tight_layout()
    
    #add inset plots
    positions = [[.75, .6, .2, .3],[.75, .2, .2, .3]]
    a = fig.add_axes(positions[0])
    b = fig.add_axes(positions[1])
    list_axes = [a,b]
    
    #prepare variables to add single-subject response error and imprecision values
    means_tmp = np.zeros((n_subs, len(selTrials)))
    sds_tmp = np.zeros((n_subs, len(selTrials)))
    
    for subi, sub in enumerate(subs):
        for i in range(len(selTrials)):
            data_tmp = selTrials[i]
            data_tmp = data_tmp[data_tmp['Subject'] == sub]
            
            means_tmp[subi,i] = np.mean(abs(data_tmp.MemItem_Repr_Error))
            sds_tmp[subi,i] = np.rad2deg(circ.std(np.deg2rad(data_tmp.MemItem_Repr_Error)))
    
    #prepare variables for mean+sem of response error and imprecision
    output_means = np.zeros((len(selTrials),2))
    output_sds = np.zeros((len(selTrials),2))
    
    for i in range(len(selTrials)):
        output_means[i,0] = np.mean(means_tmp[:,i])
        output_means[i,1] = stats.sem(means_tmp[:,i])
        
        output_sds[i,0] = np.mean(sds_tmp[:,i])
        output_sds[i,1] = stats.sem(sds_tmp[:,i])
    
    #inset-iterable single subject data
    list_ss_means_sds = [means_tmp,sds_tmp]
    
    #inset-iterable mean data
    list_means_sds = [output_means, output_sds]
    
    #inset-iterable plot formatting
    ylabels = ['Absolute\nresponse error [°]','Imprecision\n[circular SD]']
    ylims = [[4,15],[4,20]]
    yticks = [[5,15],[5,20]]
    hlines_list = [15,20]
    
    #iterate through both insets to plot them
    for p in range(len(list_axes)):
        
        #format plot
        colors = ['b','r']
        list_axes[p].spines['left'].set_color('k')
        list_axes[p].spines['bottom'].set_color('k')
        list_axes[p].spines['right'].set_visible(False)
        list_axes[p].spines['top'].set_visible(False)
        xticks_means = np.arange(2)
        list_axes[p].set_xlim([-0.5,1.5])
        list_axes[p].set_xticks([0.5])
        list_axes[p].set_xticklabels([])
        list_axes[p].tick_params(bottom=False)
        list_axes[p].set_ylim(ylims[p])
        list_axes[p].set_yticks(yticks[p])
        list_axes[p].set_ylabel(ylabels[p])
        
        #add significance in case of multiple passes plotted
        if len(selTrials) == 2:
            list_axes[p].hlines(hlines_list[p], 0, 1, colors = 'k')
            list_axes[p].text(0.3, hlines_list[p], "***", size=12, weight = 'semibold')
            
        #for each pass, plot mean, sem, and single-subject datapoints
        for i in range(len(selTrials)):
            list_axes[p].bar(xticks_means[i], list_means_sds[p][i,0], yerr = list_means_sds[p][i,1],edgecolor='k', color = colors[i], width = 0.3)
            
            jitter = rng.integers(low=-5, high=5, size=len(means_tmp))
            jitter = jitter/100
            jitter = np.expand_dims(jitter, 1)
            
            x_tmp = np.tile(xticks_means[i], (len(means_tmp), 1))+jitter
            
            list_axes[p].errorbar(x_tmp, list_ss_means_sds[p][:,i], marker = 'o', linestyle='None', markeredgecolor = colors_fillbetweens[i], c = 'None', alpha = 0.3)
    return

def plot_template_reproduction_error(data, plot_single_subjects = False):
    import numpy as np
    import scipy.stats as stats
    import pycircstat as circ
    import matplotlib.pyplot as plt
    """
    Function to plot performance measures of template reproduction.
    
    :param data: Full dataframe with all subjects' data.
    :param plot_single_subjects: Whether to plot separate plots for each subject.
    
    """
    
    rng = np.random.default_rng()
    
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    bins = np.linspace(-90, 90, 37)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    
    conditions= ['Timepoint 1\nSession 1','Timepoint 1\nSession 2',
                 'Timepoint 2\nSession 1','Timepoint 2\nSession 2']
    colors_conds = ['b','r','b','r']
    colors_fillbetweens = ['cyan','orange','cyan','orange']
    linestyles_conds = ['solid','solid','dotted','dotted']
    
    xticks_insets = [0.2,0.7,2.3,2.8]
    
    #variables to collect data for all three plots
    data_for_hist = np.zeros((len(conditions),n_subs,len(bins)-1))
    data_insetmeans = np.zeros((len(conditions),n_subs))
    data_insetsds = np.zeros((len(conditions),n_subs))
    
    for subi, sub in enumerate(subs):
        #select starts and ends of all miniblocks + relevant columns
        sel = data[(data['Subject'] == sub) & ((data['miniblock_start'] == 1) | (data['miniblock_end'] == 1))]
        sel = sel[['Templ1_R1_error', 'Templ2_R1_error',"Templ1_R2_error","Templ2_R2_error",'pass_no']]

        m = 0 #counter
                                                              
        for r in range(2): #reproduction number
            for k in range(2): #session number
                if r == 0:
                    #for first reproduction, select columns dealing with first reproduction
                    tmp = sel.iloc[:, [0,1,4]]
                elif r == 1:
                    #for second reproduction, select columns dealing with second reproduction
                    tmp = sel.iloc[:, [2,3,4]]
                
                #select rows for correct pass
                tmp = tmp[tmp['pass_no'] == (k+1)]
                
                """
                Mistake in thesis: instead of the following line (318), the plot was made with this:
                    tmp = np.concatenate((tmp.iloc[:,0].to_numpy(),tmp.iloc[:,0].to_numpy()))
                """
                
                #concatenate errors for first and second template
                tmp = np.concatenate((tmp.iloc[:,0].to_numpy(),tmp.iloc[:,1].to_numpy()))
                tmp = tmp[~np.isnan(tmp)]
                
                #compute hist over reproduction errors
                tmp_hist, _ = np.histogram(tmp, bins=bins, density=False) #compute histogram
                data_for_hist[m, subi, :] = tmp_hist / np.sum(tmp_hist)
                
                #compute imprecision
                data_insetsds[m, subi] = np.rad2deg(circ.std(np.deg2rad(tmp)))
                
                #compute absolute response error
                tmp_abs = abs(tmp)
                data_insetmeans[m, subi] = np.mean(tmp_abs)
                
                m+=1
    
    #compute mean + sem of single subject histograms of response error
    tmp_mean = np.nanmean(data_for_hist, axis=1)
    tmp_sem = stats.sem(data_for_hist, axis = 1, nan_policy = 'omit')
    
    #format plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Template reproduction error [°]', fontsize=13, fontname='Arial')
    ax.set_xticks([-90,-60,-30,0,30,60,90])
    ax.set_xlim([-90,90])
    ax.set_ylabel('Probability', fontsize=13, fontname='Arial')
    ax.set_yticks([0,0.4])
    ax.set_ylim([0,0.4])
    ax.yaxis.set_label_coords(-0.02,.5)
    ax.vlines(0, 0, .4, colors='dimgray', linestyles='dotted')
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #create insets
    positions = [[.75, .65, .2, .3],[.75, .25, .2, .3]]
    a = fig.add_axes(positions[0])
    b = fig.add_axes(positions[1])
    
    #inset-iterable plot formatting
    list_axes = [a,b]
    ylabels = ['Absolute\nresponse error [°]','Imprecision\n[circular SD]']
    ylims = [[4,30],[4,40]]
    yticks = [[5,15,30],[5,20,40]]
    hlines_list = [[20,20,22,24],[30,30,33,36]]
    
    #manually determined with statistics in R
    hlines_text = [['n.s.','**','***','***'],['n.s.','n.s.','***','*']]
    
    #iterate over both insets
    for p in range(len(list_axes)):
        #format plot
        list_axes[p].spines['left'].set_color('k')
        list_axes[p].spines['bottom'].set_color('k')
        list_axes[p].spines['right'].set_visible(False)
        list_axes[p].spines['top'].set_visible(False)
        list_axes[p].set_xlim([-0.5,3])
        list_axes[p].set_xticks([0.45,2.55])
        list_axes[p].set_xticklabels(['timepoint 1','timepoint 2'], fontsize = 7.5)
        list_axes[p].set_ylim(ylims[p])
        list_axes[p].set_yticks(yticks[p])
        list_axes[p].set_ylabel(ylabels[p])
        
        #format significance markers
        list_axes[p].hlines(hlines_list[p][0], 0.2, 0.7, colors = 'k', linewidth = 0.8)
        list_axes[p].text(0.3, hlines_list[p][1], hlines_text[p][0], size=6.5, weight = 'normal')
        list_axes[p].hlines(hlines_list[p][1], 2.3, 2.8, colors = 'k', linewidth = 0.8)
        list_axes[p].text(2.46, hlines_list[p][1]-0.1, hlines_text[p][1], size=7.5, weight = 'normal')
        list_axes[p].hlines(hlines_list[p][2], 0.2, 2.3, colors = 'k', linewidth = 0.8)
        list_axes[p].text(1.15, hlines_list[p][2]-0.1, hlines_text[p][2], size=7.5, weight = 'normal')
        list_axes[p].hlines(hlines_list[p][3], 0.7, 2.8, colors = 'k', linewidth = 0.8)
        list_axes[p].text(1.7, hlines_list[p][3], hlines_text[p][3], size=7.5, weight = 'normal')
        
    #for each condition, determine single-subject, mean, and sem of response error and imprecision and plot in respective inset    
    for condi in range(len(conditions)):
        #compute mean and sem of response errors
        overall_mean = np.mean(data_insetmeans[condi,:])
        overall_mean_sem = stats.sem(data_insetmeans[condi,:])
        
        #compute mean and sem of imprecisions
        overall_sd = np.mean(data_insetsds[condi,:])
        overall_sd_sem = stats.sem(data_insetsds[condi,:])
        
        #plot response error histograms
        ax.plot(bin_centers,tmp_mean[condi,:],color=colors_conds[condi],label=conditions[condi], linestyle = linestyles_conds[condi])
        ax.fill_between(bin_centers, tmp_mean[condi,:]-tmp_sem[condi,:],tmp_mean[condi,:]+tmp_sem[condi,:], alpha=0.8, linestyle = linestyles_conds[condi],edgecolor=colors_conds[condi], facecolor=colors_fillbetweens[condi])
    
        #plot absolute response error
        a.bar(xticks_insets[condi], overall_mean, yerr = overall_mean_sem,edgecolor='k', color = colors_conds[condi], width = 0.3)
        #plot imprecision
        b.bar(xticks_insets[condi], overall_sd, yerr = overall_sd_sem,edgecolor='k', color = colors_conds[condi], width = 0.3)

        jitter = rng.integers(low=-5, high=5, size=len(data_insetmeans[condi,:]))
        jitter = jitter/100
        jitter = np.expand_dims(jitter, 1)
        
        x_tmp = np.tile(xticks_insets[condi], (len(data_insetmeans[condi,:]), 1))+jitter
        
        #plot single-subject datapoints in insets
        a.errorbar(x_tmp, data_insetmeans[condi,:], marker = 'o', linestyle='None', markeredgecolor = colors_fillbetweens[condi], c = 'None', alpha = 0.3)
        b.errorbar(x_tmp, data_insetsds[condi,:], marker = 'o', linestyle='None', markeredgecolor = colors_fillbetweens[condi], c = 'None', alpha = 0.3)

    ax.legend(loc=3, prop={'size': 8.5}, frameon = False, bbox_to_anchor=(0, 0.55))
    fig.tight_layout()
    return

def plot_active_latent_together(data, plot_single_subjects = False, response = 'ccw', binsize = 10, fit_fromto = 20):
    import numpy as np   
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from MemTempl_analysis_functions_basic import cwccw_response_ratio_distance
    from MemTempl_analysis_functions_fits import glm_fit
    """
    Function to plot judgment task responses depending on template-Gabor angle difference.
    
    :param data: Full dataframe with all subjects' data.
    :param plot_single_subjects: Whether to plot separate plots for each subject.
    :param response: Whether to plot clockwise or counterclockwise response ratio on y-axis (mirrors plot if changed)
    :param binsize: Size of bins in which angular distances are pooled [°]
    :param fit_fromto: Absolute angular distance between which and 0 a binomial curve is fitted to the plot.
    
    """
    
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    conds = ['active','latent']
    labels = ['active template','latent template']
    
    n_cond = len(conds)
    
    """
    ===
    Prepare variable to collect linear fit coefficients for subjects and conditions
    ===
    
    data_slopes = np.zeros((n_subs,n_cond))
    colorsfit = ['orange','cyan']
    """
    
    colors = ['r','b']
    colorsfillbetween = ['darkorange','cyan']
    
    #determine binnumber from size and create bins
    binnumber = 180/binsize
    bins = np.linspace(-90,(90-binsize),int(binnumber))
    
    #determine selection of bins to be used for (glm) fit from fit_fromto
    sel_idx = np.where((bins>-fit_fromto)&(bins< fit_fromto))
    bin_sel = [np.min(sel_idx),np.max(sel_idx)]
    
    #prepare variable to collect single-subject response ratio for each condition
    data_tmp = np.zeros((n_subs, n_cond, len(bins)))
        
    for subi, sub in enumerate(subs):
        if plot_single_subjects:
            gif = plt.figure()
            axo = gif.add_subplot()
            axo.set_title(str(sub))
        
        #select single subject data
        sel = data[data['Subject'] == sub].reset_index(drop=True)
        
        #prepare variable to store single subject's data for both conditions
        both_cond = np.zeros((n_cond,len(bins)))
        
        for condi, cond in enumerate(conds):
            data_tmp[subi,condi,:] = cwccw_response_ratio_distance(sel,conds[condi], response, binsize)
            both_cond[condi, :] = data_tmp[subi,condi,:]
            
            """
            ===
            Plot linear fits over middle parts of curves
            ===
            
            coef = np.polyfit(bins[bin_sel[0]:bin_sel[1]],data_tmp[subi,condi,bin_sel[0]:bin_sel[1]] , 1)
            data_slopes[subi,condi] = coef[0]
            poly1d_fn = np.poly1d(coef) 
            """
            if plot_single_subjects:
                axo.plot(bins,data_tmp[subi,condi,:],marker='o',color=colors[condi],label=labels[condi],linestyle='-')   
                """
                ===
                Plot linear fits over middle parts of curves
                ===
                
                axo.plot(bins[bin_sel[0]:bin_sel[1]], poly1d_fn(bins[bin_sel[0]:bin_sel[1]]), c = colorsfit[condi]) #'--k'=black dashed line, 'yo' = yellow circle marker 
                """
    
    #prepare variable to collect mean and sem of single-subject distributions for each condition
    data_mean = np.zeros((n_cond, len(bins)))
    data_sem = np.zeros((n_cond, len(bins)))

    #calculate mean and sem for each condition
    mean_tmp = np.nanmean(data_tmp, axis=0)
    sem_tmp = stats.sem(data_tmp, axis = 0)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    #plot each condition sequentially
    for condi in range(n_cond):
        data_mean[condi,:] = mean_tmp[condi]
        data_sem[condi, :] = sem_tmp[condi]
        
        """
        ===
        Plot linear fits over middle parts of curves
        ===
        coef = np.polyfit(bins[bin_sel[0]:bin_sel[1]],data_mean[condi,bin_sel[0]:bin_sel[1]] , 1)
        poly1d_fn = np.poly1d(coef) 
        """

        for subi, sub in enumerate(subs):
            ax.plot(bins+(binsize/2),data_tmp[subi,condi,:],marker='o',color=colors[condi],linestyle='None', alpha=0.1) 
        
        ax.plot(bins+(binsize/2),data_mean[condi,:],marker='o',color=colors[condi],label=labels[condi],linestyle='None')   
        ax.fill_between(bins+(binsize/2), data_mean[condi,:]-data_sem[condi,:], data_mean[condi,:]+data_sem[condi,:],edgecolor=colorsfillbetween[condi], facecolor=colorsfillbetween[condi], alpha = 0.8)
        
    #fit GLM binomial to slope
    _,_,y_preds = glm_fit(data_mean, bins+(binsize/2),bin_sel)
    
    #plot binomials
    for condi in range(n_cond):
        ax.plot(bins[bin_sel[0]:bin_sel[1]]+(binsize/2), y_preds[condi], c='k')

    #format plot
    x_horiz = np.linspace(-90,90,100)
    y_vert = np.linspace(0,1,50)
    x_vert = np.repeat(0, len(y_vert))
    y_horiz = np.repeat(0.5, len(x_horiz))    
    plt.plot(x_horiz, y_horiz,linestyle = 'dotted', color="#808080", alpha=0.7)
    plt.plot(x_vert, y_vert,linestyle = 'dotted', color="#808080", alpha=0.7)
    ax.set_xlabel( r'$\longleftarrow$' + " Gabor counterclockwise     |     Gabor clockwise " + r'$\longrightarrow$', fontsize = 10)
    ax.xaxis.set_label_coords(.443, -.1)
    if response == 'ccw':
        resplabel = "counterclockwise"
    elif response == 'cw':
        resplabel = "clockwise"
    ax.set_ylabel("Ratio of " + str(resplabel) + " responses", fontsize = 10)
    yticks = [0,0.5,1.0]
    ax.set_yticks(yticks)
    ax.set_ylim([-0.01, 1.01])
    ax.set_xticks([-90,-60,-30,0,30,60,90])
    ax.set_xlim([-90,90])
    #title = 'Probability of ' + str(response) + ' response active vs latent'
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.legend(loc=3, prop={'size': 9}, frameon = False, bbox_to_anchor=(0.65,0.75))
    return

def plot_judgment_results(judgment_statistics_data):
    import scipy.stats as stats
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Function to plot judgment task performance by color and session.
    
    :param judgment_statistics_data: Prepared dataframe with judgment task performances under different conditions.
    
    """
    
    subs = np.unique(judgment_statistics_data.Sub_ID)
    n_subs = len(subs)

    rng = np.random.default_rng()
    
    #conditions as string only for better readability - can also be np.arange(4)
    conditions = ['Session 1, difficult colors','Session 2, difficult colors',
                  'Session 1, easy colors','Session 2, easy colors']
    
    cond_colors = ['b','r','b','r']
    colors_singlesubjects = ['cyan','orange','cyan','orange']

    labels = ['Session 1','Session 2','_nolegend_','_nolegend_']
    
    #set up plot
    fig = plt.figure()
    ax = fig.add_subplot()
    xticks = [0.2, .7, 1.7,2.2]
    
    #dataframe contains more conditions than are relevant to plot here; draw them out of it into new variable
    #relevant conditions (in this case, color and session) are determined by ANOVA in R
    data_coll = np.zeros((len(conditions),n_subs))
    
    for subi, sub in enumerate(np.unique(judgment_statistics_data.Sub_ID)):
        m = 0
        for j in np.unique(judgment_statistics_data.Session_Number):
            for k in np.unique(judgment_statistics_data.Template_Color_0hard_1easy):
                #for each subject, select relevant conditions
                sel = judgment_statistics_data[(judgment_statistics_data['Sub_ID'] == sub) & (judgment_statistics_data['Session_Number'] == j)&(judgment_statistics_data['Template_Color_0hard_1easy'] == k)]
                
                #insert single-subject mean over all performances meeting the conditions in current iteration
                data_coll[m, subi] = np.nanmean(sel.perf_cwccw)
                m += 1
    
    #make variables to store mean and sem over all subjects for each condition
    means = np.zeros((len(conditions)))
    sems = np.zeros((len(conditions)))
    
    for condi in range(len(conditions)):
        #determine mean and sem for given condition
        means[condi] = np.mean(data_coll[condi,:])
        sems[condi] = stats.sem(data_coll[condi,:])
        
        #plot mean and sem as bar plots
        ax.bar(xticks[condi], means[condi], yerr = sems[condi],edgecolor='k', color = cond_colors[condi], width = 0.3, label = labels[condi])
    
        #plot single-subject performances over barplots
        jitter = rng.integers(low=-5, high=5, size=len(data_coll[condi,:]))
        jitter = jitter/100
        jitter = np.expand_dims(jitter, 1)
        x_tmp = np.tile(xticks[condi], (len(data_coll[condi,:]), 1))+jitter
    
        ax.errorbar(x_tmp, data_coll[condi,:], marker = 'o', linestyle='None', markeredgecolor = colors_singlesubjects[condi], c = 'None', alpha = 0.3)
      
    #format plot
    ax.set_xlim([-.2, 2.6])
    ax.set_ylim([0.6, 1])
    ax.set_xticks([0.45,1.95])
    ax.set_xticklabels(['Difficult colors', 'Easy colors'])
    ax.set_ylabel("Ratio of correctly answered trials")
    ax.set_yticks([0.6,1])
    ax.yaxis.set_label_coords(-0.02,.5)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc=3, prop={'size': 8.5}, frameon = False, bbox_to_anchor=(0, 0.85))
    
    #format significance markers (MISTAKE HERE, CORRECT IT)
    ax.hlines(0.9, 0.2, 0.7, colors = 'k', linewidth = 0.8)
    ax.text(0.42, 0.9, "**", size=12, weight = 'normal')
    
    ax.hlines(0.9, 1.7, 2.2, colors = 'k', linewidth = 0.8)
    ax.text(1.92, 0.905, "n.s.", size=11, weight = 'normal')
    
    ax.hlines(0.92, 0.2, 1.7, colors = 'k', linewidth = 0.8)
    ax.text(0.95, 0.925, "n.s.", size=11, weight = 'normal')
    
    ax.hlines(0.94, 0.7, 2.2, colors = 'k', linewidth = 0.8)
    ax.text(1.45, 0.945, "n.s.", size=11, weight = 'normal')
    return

def prepare_statistics_MemItem_Repr_performance(data, save = False):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from MemTempl_analysis_config import path_results
    import pycircstat as circ
    import scipy.stats as stats
    """
    Function to create Gabor reproduction performance dataframe to export for statistics.
    
    :param data: Full dataframe with all subjects' data.
    :param save: Whether to save dataframe as csv.
    
    """

    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    ##prepare variables to store single-subject datapoints
    means_tmp = np.zeros((n_subs,2))
    sds_tmp = np.zeros((n_subs,2))
    subcodes = np.zeros((n_subs,1))
    
    ##get single-subject datapoints
    for subi, sub in enumerate(subs):
        for passi in range(2):
            #make subjectcode shorter
            subjcode = int(str(sub[-3:]))
            #sds_tmp[subi,2] = subjcode
            subcodes[subi,:] = subjcode
            
            #select subject- and pass-specific trials
            data_tmp = data[data['Subject'] == sub]
            data_tmp = data_tmp[data_tmp['pass_no'] == (passi+1)]
            
            #for each pass, calculate and store absolute reproduction error and circular standard deviations
            means_tmp[subi,passi] = np.mean(abs(data_tmp.MemItem_Repr_Error))
            sds_tmp[subi,passi] = np.rad2deg(circ.std(np.deg2rad(data_tmp.MemItem_Repr_Error)))
    
    ##calculate overall mean+sem
    output_means = np.zeros((2,2))
    output_sds = np.zeros((2,2))
    
    for i in range(2):
        output_means[i,0] = np.mean(means_tmp[:,i])
        output_means[i,1] = stats.sem(means_tmp[:,i])
        
        output_sds[i,0] = np.mean(sds_tmp[:,i])
        output_sds[i,1] = stats.sem(sds_tmp[:,i])
    
    ##build dataframe
    #reshape means of both passes to stack on top of each other
    means = np.concatenate((means_tmp[:,0],means_tmp[:,1]))
    means = means.reshape(len(means),1)
    
    #reshape circular standard deviations of both passes to stack on top of each other
    sdss = np.concatenate((sds_tmp[:,0],sds_tmp[:,1]))
    sdss = sdss.reshape(len(sdss),1)

    #copy list of subjectcodes and stack copy on top of original
    #subjcolumn = np.tile((sds_tmp[:,2]),2)
    subjcolumn = np.tile((subcodes[:,0]),2)
    subjcolumn = subjcolumn.reshape(len(subjcolumn),1)

    #create column denoting pass-number
    passcolumn = np.repeat((0,1),n_subs)
    passcolumn = passcolumn.reshape(len(passcolumn),1)

    #add all columns together
    tmp = np.hstack((means, sdss))
    tmp = np.hstack((tmp, subjcolumn))
    tmp = np.hstack((tmp, passcolumn))
    
    #turn into pandas dataframe
    exp_data = pd.DataFrame(tmp,columns=['MemItem_Rep_Error','circ_sd','Subj_ID','session'])
    
    #export to csv
    if save:
        R_dataname = 'MemItem_Rep_performance.csv'
        filename = Path(path_results / R_dataname)
        exp_data.to_csv(filename,columns = ['MemItem_Rep_Error','circ_sd','Subj_ID','session'], index=False, header=True)
        
    return exp_data, output_means, output_sds

def prepare_statistics_Template_Repr_performance(data, save = False):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from MemTempl_analysis_config import path_results
    import pycircstat as circ
    """
    Function to create Template reproduction performance dataframe to export for statistics.
    
    :param data: Full dataframe with all subjects' data.
    :param save: Whether to save dataframe as csv.
    
    """

    subs = np.unique(data.Subject)
    
    #colors separated according to post-experimental questionnaires:
        #first list: difficult colors (blues, pink&purple); second list: easy colors (the rest)
    colors = [[174,197,184,207,194,217,204,227,274,297],[4,27,14,37,24,47,34,57,54,77,254,277,264,287]] 
    
    #variable to collect datapoints for all subjects
    coll_means = []
    
    for subi, sub in enumerate(subs):
        #prepare variable to contain all data for one subject:
            #one mean and circular standard deviation for each of 16 conditions, 4 variables, and one column for subject-ID
        tmp_sub_mean = np.zeros((16,7))
        
        #select template-reproduction trials for specific subject
        sel = data[(data['Subject'] == sub) & ((data['miniblock_start'] == 1) | (data['miniblock_end'] == 1))]
        
        m = 0
        #prepare for ANOVA: separate performance measures for trials sharing a combination of conditions as follows                                                        
        for i in range(2): #template number (1 or 2 being reproduced)
            for r in range(2): #reproduction number (reproduction happening at beginning or end of miniblock)
                for k in range(2): #session number (reproductions in first or second session)
                    for c in range(len(colors)): #color (reproduction of template shown in difficult or easy color)
                        #select template number
                        if i == 0:
                            tmp = sel[['Templ1_R1_error', "Templ1_R2_error",'pass_no','Templ1_color']]
                        elif i == 1:
                            tmp = sel[['Templ2_R1_error', "Templ2_R2_error",'pass_no','Templ2_color']]
                        
                        #select reproduction number
                        tmp = tmp.iloc[:, [r, 2,3]]
                        
                        #select session number
                        tmp = tmp[tmp['pass_no'] == (k+1)]
                        
                        #select colors
                        tmp = tmp[tmp.iloc[:,-1].isin(colors[c])]
                        
                        #column 0 now has the right data
                        tmp = tmp.iloc[:,0].values
                        tmp = tmp[~np.isnan(tmp)]
                        
                        tmp_abs = abs(tmp)
                        
                        tmp_sub_mean[m,:6] = np.mean(tmp_abs), np.rad2deg(circ.std(np.deg2rad(tmp))),i, r, k, c
                        m += 1
        
        tmp_sub_mean[:,6] = subi
        
        #add to collect all subjects
        coll_means.append(tmp_sub_mean)
    
    #turn list of collected subjects into array
    means = np.concatenate((coll_means))
    
    #turn array into pandas dataframe
    exp_data = pd.DataFrame(means,columns=['Template_Rep_Error','Template_Rep_sd','Template_Number','Reproduction_Number','Session_Number','Template_Color_0hard_1easy','Sub_ID'])
    
    if save:
        R_dataname = 'Template_Rep_performance.csv'
        filename = Path(path_results / R_dataname)
        exp_data.to_csv(filename,columns = ['Template_Rep_Error','Template_Rep_sd','Template_Number','Reproduction_Number','Session_Number','Template_Color_0hard_1easy','Sub_ID'], index=False, header=True)
            
    return exp_data

def prepare_statistics_Template_judgment(data, save = False):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from MemTempl_analysis_config import path_results
    """
    Function to create judgment task performance dataframe to export for statistics.
    
    :param data: Full dataframe with all subjects' data.
    :param save: Whether to save dataframe as csv.
    
    """

    subs = np.unique(data.Subject)

    #colors separated according to post-experimental questionnaires:
        #first list: difficult colors (blues, pink&purple); second list: easy colors (the rest)
    colors = [[174,197,184,207,194,217,204,227,274,297],[4,27,14,37,24,47,34,57,54,77,254,277,264,287]] 
    
    #variable to collect datapoints for all subjects
    coll_perfs = []
    
    for subi, sub in enumerate(subs):
        #prepare variable to contain all data for one subject:
            #one performance measure in percent for each of 8 conditions, 3 variables, and one column for subject-ID
        tmp_sub_perf = np.zeros((8,5))
        
        sel = data[data['Subject'] == sub]
        m = 0
        
        #prepare for ANOVA: separate performance measures for trials sharing a combination of conditions as follows                                                                                                                
        for i in range(2): #template number (1 or 2 being used for judgment task)
            for k in range(2): #session number (1st or 2nd pass)
                for c in range(len(colors)): #judgment task using template shown in easy or difficult color
                    
                    #select trials according to template number
                    tmp = sel[sel['Active'] == i+1]
                    if i == 0:
                        tmp = tmp[['performance_cwccw','pass_no','Templ1_color']]
                    elif i == 1:
                        tmp = tmp[['performance_cwccw','pass_no','Templ2_color']] #select template number
                    
                    #select trials from correct session
                    tmp = tmp[tmp['pass_no'] == (k+1)]
                    
                    #select trials in correct color
                    tmp = tmp[tmp.iloc[:,-1].isin(colors[c])]
                    
                    #extract performance (correct or wrong) for each of the selected trials
                    tmp = tmp.iloc[:,0].values
                    tmp = tmp[~np.isnan(tmp)]
                    
                    #calculate percentage of correct responses as measure of performance
                    tmp_sub_perf[m,:4] = np.count_nonzero(tmp == 1)/len(tmp), i, k, c
                    
                    m += 1
        
        tmp_sub_perf[:,4] = subi
        
        #add to collect all subjects
        coll_perfs.append(tmp_sub_perf)
        
    #turn list of collected subjects into array
    perfs = np.concatenate((coll_perfs))
    
    #turn array into pandas dataframe
    exp_data = pd.DataFrame(perfs,columns=['perf_cwccw','Template_Number','Session_Number','Template_Color_0hard_1easy', 'Sub_ID'])
    
    if save:
        R_dataname = 'Template_Judg_performance.csv'
        filename = Path(path_results / R_dataname)
        exp_data.to_csv(filename,columns = ['perf_cwccw','Template_Number','Session_Number','Template_Color_0hard_1easy', 'Sub_ID'], index=False, header=True)
            
    return exp_data

def prepare_statistics_serDep(data_clean, split, n_permutations, save = False):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from MemTempl_analysis_functions_basic import return_modelfreemeasure_nonsmoothed
    from MemTempl_analysis_config import path_results
    from MemTempl_analysis_functions_fits import permute_modfree_SD_significance
    """
    Function to create model-free serial dependence measure dataframe to export for statistics.
    
    :param data_clean: Full dataframe with all subjects' cleaned data (cleaned by remove_outliers_SD() function).
    :param split: List of degree values at which to split data sequences. Dependence will be determined separately for each sequence.
    :param n_permutations: Number of permutations to determine significance of single-subject dependence measures.
    :param save: Whether to save dataframe as csv.
    
    """
    SD_types = ['SD_classic', 'SD_resp', 'SD_actTempl','SD_latTempl']
    subs = np.unique(data_clean.Subject)
    
    subrange = np.arange(len(subs))
    reprange = np.arange(len(split))
    
    #Variable to collect full column of dependence measures for each independent variable in SD_types.
    collect_realmeasures = []
    
    #Iterate through independent variables and build column fit for R analysis.
    for typi in range(len(SD_types)):
        #Store one dependence measure for each section of data, for each subject, in one column.
        real_measures = np.zeros((len(subs),len(split)))
        real_measures[:,:] = return_modelfreemeasure_nonsmoothed(data_clean, SD_types[typi], split = split, meanormedian = 'median', degreeorradian= 'degree')
        real_measures = real_measures.T.reshape(-1, 1)
        
        #Store significance for each section of data, for each subject, in one column.
        permuted_sds = np.zeros((len(subs), len(split)))
        permuted_sds[:,:] = permute_modfree_SD_significance(data_clean, SD_types[typi], split = split, n_permutations = n_permutations)
        perm_signif = permuted_sds.T.reshape(-1, 1)
        
        # sectionsignif = []
        # sectionsignif.append(permuted_sds[:,0])
        # sectionsignif.append(permuted_sds[:,1])
        
        # perm_signif = np.concatenate((sectionsignif))
        # perm_signif = perm_signif.reshape(len(perm_signif),1)
        
        #Number sections depending on split.
        sectionno = np.repeat(reprange,len(subs))
        sectionno = sectionno.reshape(len(sectionno),1)
        
        #Number independent variable of current iteration.
        typno = np.repeat(typi,len(real_measures))
        typno = typno.reshape(len(typno),1)
        
        #Number subjects.
        subno = np.tile(subrange,len(split))
        subno = subno.reshape(len(subno),1)
        
        #Combine all established columns.
        real_measures = np.hstack((real_measures,perm_signif))
        real_measures = np.hstack((real_measures,sectionno))
        real_measures = np.hstack((real_measures,typno))
        real_measures = np.hstack((real_measures,subno))
        
        #Add to collecting variable.
        collect_realmeasures.append(real_measures)
    
    #Turn list into array and then into dataframe.
    realmeasures_array = np.concatenate((collect_realmeasures), axis=0)
    exp_data = pd.DataFrame(realmeasures_array,columns=['SD_measure','significance','split','type','Sub_ID'])
    
    if save:
        R_dataname = "Modelfree_dependence_measure.csv"
        filename = Path(path_results / R_dataname)
        exp_data.to_csv(filename,columns = ['SD_measure','significance','split','type','Sub_ID'], index=False, header=True)
        
    return exp_data

def remove_outliers_SD(data, cutoff = 3, qualCheck = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from MemTempl_analysis_functions_basic import get_CircError
    """
    Function to remove trials with outlier response errors and demeaned response errors (as in Fischer 2014).
    
    Returns adjusted data frame.
    
    :param data: Full dataframe with all subjects' data.
    :param cutoff: Determines number by which standard deviation of response errors is multiplied to reach threshold for removal of trials.
    :param qualCheck: Whether to plot distribution of response errors (should be clustered around the diagonal).
    
    """
    
    subs = np.unique(data.Subject)
    
    #new variable to collect all subjects' clean data
    data_tmp = []
    
    #create new columns that will be altered to carry demeaned/nan'd data points
    data['Resp_error_norm'] = data.MemItem_Repr_Error.values
    data['Corr_resp_error_norm'] = data.Corr_MemItem_Resp_Error.values
    
    #prepare variable to carry percentage of removed trials for each subject
    perc_removed = {}
    
    for subi, sub in enumerate(subs):
        perc_removed[sub] = np.zeros((2))
        
        if qualCheck:
            fig = plt.figure()
            ax = fig.add_subplot()
        
        sel = data[data['Subject'] == sub].reset_index(drop=True)
        sel_clean = sel.copy(deep=True)
        
        #get circular mean and standard deviation of Gabor reproduction
        tmp_av_err, sd_err = get_CircError(sel_clean.MemItem_Repr_Error.values,directionality = 1)
        if tmp_av_err > 300:
            av_err = tmp_av_err-360
        else:
            av_err = tmp_av_err
        
        #determine cutoff to remove trials as cutoff*standard deviation
        cutoff_sub = cutoff*sd_err
        
        #nan response in every trial where the response error is beyond the cutoff
        if True in np.unique((sel_clean.Resp_error_norm > np.mod(av_err+cutoff_sub, 90)) |
        (sel_clean.Resp_error_norm < np.mod(av_err-cutoff_sub, -90))):
            sel_clean.Resp_error_norm[(sel_clean.Resp_error_norm > np.mod(av_err+cutoff_sub, 90)) |
                               (sel_clean.Resp_error_norm <  np.mod(av_err-cutoff_sub, -90))] = np.nan
        
        #determine the number of trials that were nan'd in the previous step
        sel_removed = sel.copy(deep=True)
        sel_removed = sel[np.isnan(sel_clean.Resp_error_norm)]
        
        #store this number as a percentage of all trials
        perc_removed[sub][0] = (len(sel_removed)/len(sel_clean))*100
        #and store this number as an absolute number
        perc_removed[sub][1] = len(sel_removed)
    
        #plot all reproduction errors along a diagonal (removed and not-removed in different colors)
        if qualCheck:
            ax.set_title(str(sub))
            ax.scatter(sel_clean.MemItem_angle-90, sel_clean.MemItem_Repr-90, color='b') #clean data
            ax.scatter(sel_removed.MemItem_angle-90,sel_removed.MemItem_Repr-90,color='r')
            ax.plot(np.linspace(-90, 90, 1000), np.linspace(-90, 90, 1000), color='k', alpha=.8)
        
        #calculate new mean response error (of cleaned data)
        tmp_av_err, _ = get_CircError(sel_clean.Resp_error_norm.values, directionality=1) 
        if abs(tmp_av_err) > 300: 
            new_av_err = tmp_av_err-360 
        else:
            new_av_err = tmp_av_err
        
        #subtract this new response error from all leftover response errors and add as new column (nan'd and demeaned)
        Resp_error_demeaned = sel_clean.Resp_error_norm.values
        Resp_error_demeaned = Resp_error_demeaned - new_av_err
    
        sel_clean['Resp_error_demeaned'] = Resp_error_demeaned
        
        #add single-subject data to new collecting variable
        data_tmp.append(sel_clean)
    
    #turn collection of cleaned data into dataframe
    data_new = pd.concat(data_tmp)
    
    ##do the same thing again but for the oblique-bias corrected response errors
    #adding that to the cleaned dataframe
    data_tmp = []
    
    for subi, sub in enumerate(subs):
        if qualCheck:
            fig = plt.figure()
            ax = fig.add_subplot()
            
        sel = data_new[data_new['Subject'] == sub].reset_index(drop=True)
        sel_clean = sel.copy(deep=True)
        
        tmp_av_err, sd_err = get_CircError(sel_clean.Corr_MemItem_Resp_Error.values,directionality = 1)
        if tmp_av_err > 300:
            av_err = tmp_av_err-360
        else:
            av_err = tmp_av_err
        
        cutoff_sub = cutoff*sd_err
        
        if True in np.unique((sel_clean.Corr_resp_error_norm > np.mod(av_err+cutoff_sub, 90)) |
        (sel_clean.Corr_resp_error_norm < np.mod(av_err-cutoff_sub, -90))):
            sel_clean.Corr_resp_error_norm[(sel_clean.Corr_resp_error_norm > np.mod(av_err+cutoff_sub, 90)) |
                               (sel_clean.Corr_resp_error_norm <  np.mod(av_err-cutoff_sub, -90))] = np.nan
        
        sel_removed = sel.copy(deep=True)
        sel_removed = sel[np.isnan(sel_clean.Corr_resp_error_norm)]
        
        if qualCheck:
            ax.set_title(str(sub))
            ax.scatter(sel_clean.MemItem_angle-90, sel_clean.MemItem_Repr-90, color='b') #clean data
            ax.scatter(sel_removed.MemItem_angle-90,sel_removed.MemItem_Repr-90,color='r')
            ax.plot(np.linspace(-90, 90, 1000), np.linspace(-90, 90, 1000), color='k', alpha=.8)
        
        
        tmp_av_err, _ = get_CircError(sel_clean.Corr_resp_error_norm.values, directionality=1) 
        if abs(tmp_av_err) > 300: 
            new_av_err = tmp_av_err-360 
        else:
            new_av_err = tmp_av_err
        
        Corr_resp_error_demeaned = sel_clean.Corr_resp_error_norm.values
        Corr_resp_error_demeaned = Corr_resp_error_demeaned - new_av_err
    
        sel_clean['Corr_resp_error_demeaned'] = Corr_resp_error_demeaned
        
        data_tmp.append(sel_clean)
        
    data_clean = pd.concat(data_tmp)
    
    return data_clean, perc_removed

def plotMovingAverage(data, bin_width, SD_type, plot_at_all = True,
                      plot_single_subjects = False, return_single_subjects = True, 
                      mean_type = 'cross', passes = 1, n_back = 4,
                      split = [90]):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pycircstat as circ
    import scipy.stats as stats
    from MemTempl_analysis_functions_basic import pulldata
    """
    Function to plot single-subject and/or mean (using pooled or non-pooled data) dependencies 
    of Gabor reproduction error on orientation differences between stimuli of interest.

    Returns adjusted data frame.
    
    :param data: Full dataframe with all subjects' cleaned data (cleaned by remove_outliers_SD() function).
    :param bin_width: Averaging window for angular distances.
    :param SD_type: Which dependence to plot.
    :param plot_at_all: Whether to return any plots.
    :param plot_single_subjects: Whether to plot single subjects separately.
    :param return_single_subjects: Whether to return dependency curve data for single subjects (without necessarily plotting them).
    :param mean_type: "cross" to return mean + sem of all single subject data. "pool" to return mean of pooled super-subject.
    :param passes: Whether to plot both sessions pooled (1) or separately (2)
    :param n_back: For dependence on other Gabor orientations than n-1.
    :param split: Whether to plot yellow vertical lines denoting split SD measures.
    
    """

    #determine if plot starts at 0 or 90
    first_spot = 90
    if SD_type == 'SD_currori':
        first_spot = 0
        
    #determine, if there are going to be plots, what the x-range is going to be
    #usually -90 to 90, but for a specific case 0 to 90
    if plot_at_all:
        pass_colors = ['b','r']
        fill_colors = ['r','b']
        x_range = np.linspace(-90,90,181)
        xticks = [-90, -45, 0, 45, 90]
        
        if SD_type == 'SD_currori':
            x_range = np.linspace(0,180,181)
            xticks = [0,45,90,135,180]
    
    #determine if passes are going to be computed separately or pooled
    if passes == 2:
        n_pass = 2
    else:
        n_pass = 1
        
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    
    data_it = []
    
    #determine if single subject data are needed (for cross mean or because single subjects are going to be plotted):
    if mean_type == 'cross' or return_single_subjects: #single subjects needed
        return_single_subjects = True
        
        #prepare single-subject output array
        data_curve_ss = np.zeros((n_subs, n_pass, 181))
        
        #extract single-subject datasets to iterate over them
        for subi, sub in enumerate(subs):
            #variable to store single-subject data of either one pooled or two separate passes
            tmp_sub = []
            
            tmp = data[data['Subject']==sub].reset_index(drop=True)
            
            if n_pass == 2:
                tmp_sub.append(tmp[tmp['pass_no']==1].reset_index(drop=True))
                tmp_sub.append(tmp[tmp['pass_no']==2].reset_index(drop=True))
            else:
                tmp_sub.append(tmp)
                
            data_it.append(tmp_sub)     
            
    if mean_type == 'pool': #if the pooled mean is asked for, subjects are not considered separately
        tmp_mean = []
        if n_pass == 2:
            tmp_mean.append(data[data['pass_no'] == 1].reset_index(drop=True))
            tmp_mean.append(data[data['pass_no'] == 2].reset_index(drop=True))
        else:
            tmp_mean.append(data)
        
        data_it.append(tmp_mean)
    
        if not return_single_subjects:
            data_curve_ss = 'No single subjects computed'
    
    #prepare mean output array
    data_curve_mean = np.zeros((n_pass,181))
    
    plot = False

    if plot_single_subjects:
        titles = list(subs)
        titles.append('all')
        titles = np.asarray(titles)

    else:
        titles = ['all']
        titles = titles * 30
    
    
    for subi, data in enumerate(data_it):
        if plot_at_all:
            if plot_single_subjects or len(data_it) == 1 or subi >= n_subs:
                fig = plt.figure(figsize=[12, 10],dpi=300)
                ax = fig.add_subplot()
                plot = True
                
            elif not plot_single_subjects:
                plot = False
        else:
            plot = False
            
        for passi in range(n_pass):
            x, titlename,_ ,xlab= pulldata(data[passi], SD_type, n_back = n_back, subject = titles[subi])
            title_sub = titlename + str(titles[subi])
            title_all = titlename

            if SD_type == 'SD_resp' or SD_type == 'SD_currori':
                y = data[passi].Corr_resp_error_demeaned.values
            else:
                y = data[passi].Resp_error_demeaned.values
            
            y[np.isnan(x)] = np.nan
            x[np.isnan(y)] = np.nan
            
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            
            #Pad the data (to be able to smooth even those data at the 'edges' of the circle)
            x_padded = x-180
            x_padded = np.append(x_padded, x)
            x_padded = np.append(x_padded, x+180)
            
            y_padded = np.tile(y, 3)
            
            #Smooth the data (aka, take the mean of the data) within a given bin
            data_smoothed = np.zeros(181) #181
            
            for bini in range(0, 181): #181
                range_tmp = (np.array((1, bin_width)) - np.floor(bin_width/2)-1-first_spot+(bini)) #90
                data_smoothed[bini]=np.rad2deg(circ.mean(np.deg2rad(y_padded[(x_padded >= range_tmp[0]) & (x_padded <= range_tmp[1])])))
                            
            data_smoothed = np.mod(data_smoothed+90, 180)-90
            
            if return_single_subjects:
                if subi < n_subs:
                    data_curve_ss[subi,passi,:] = data_smoothed
            elif len(data_it) == 1:
                data_curve_mean[:,:] = data_smoothed   
            
            if plot:
                ax.plot(x_range, data_smoothed, c= pass_colors[passi])
                ax.hlines(0, -90, 90, colors='dimgray', linestyles='dotted')
                ax.vlines(0, -9, 9, colors='dimgray', linestyles='dotted')
                
                for spliti in split: #plot illustrative orange vertical lines
                    ax.vlines(0, -9, 9, colors='orange', linestyles='dotted')
                    ax.vlines(spliti-5, -9, 9, colors='orange', linestyles='dotted')
                    ax.vlines(-spliti+5, -9, 9, colors='orange', linestyles='dotted')
                
                ax.set_yticks([-2,0,2])
                ax.set_ylim([-2,2])
                ax.yaxis.set_label_coords(-0.05,.5)
                
                ax.set_xticks(xticks)
                ax.set_xlim([xticks[0],xticks[-1]])
                
                ax.set_title(title_sub)
    
    if mean_type == 'cross':
        y_data = np.mean(data_curve_ss, axis = 0)
        data_curve_mean[:,:] = y_data
        y_sem = stats.sem(data_curve_ss, axis = 0)
        
        if plot_at_all:
            gif = plt.figure()
            axo = gif.add_subplot()
            
            if n_pass == 2:
                pass_labels = ['first pass','second pass']
            else:
                pass_labels = ['all trials']
            
            for passi in range(n_pass):
                axo.plot(x_range, y_data[passi], marker=None, color=pass_colors[passi], label = pass_labels[passi])
                axo.fill_between(x_range, y_data[passi]-y_sem[passi], y_data[passi]+y_sem[passi], color = fill_colors[passi],alpha = 0.5)
                
            axo.hlines(0, -120, 120, colors='dimgray', linestyles='dotted')
            axo.vlines(0, -9, 9, colors='dimgray', linestyles='dotted')
            axo.spines['left'].set_color('k')
            axo.spines['bottom'].set_color('k')
            axo.spines['right'].set_visible(False)
            axo.spines['top'].set_visible(False)
            axo.yaxis.set_label_coords(-0.04,.5)
            
            for spliti in split: #plot illustrative orange vertical lines
                axo.vlines(0, -9, 9, colors='orange', linestyles='dotted')
                axo.vlines(spliti-1, -9, 9, colors='orange', linestyles='dotted')
                axo.vlines(-spliti+1, -9, 9, colors='orange', linestyles='dotted')
            
            axo.set_yticks([-2,0,2])
            axo.set_ylim([-2,2])
            axo.set_ylabel("Response bias [°]", fontsize = 11)
            
            axo.set_xticks(xticks)
            axo.set_xlim([xticks[0],xticks[-1]])
            axo.set_xlabel(xlab, fontsize=11)
            
            axo.set_title(title_all)
            gif.legend(loc='lower right')
    
    return data_curve_ss, data_curve_mean, axo, xlab

def plot_mean_modelfree_dependence(modelfree_dependence_statistics_data, split, title = False, plot_singlesubject = False, savefig = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from MemTempl_analysis_config import path_results
    """
    Function to plot mean model-free dependence measure for multiple dependencies.
    
    :param modelfree_dependence_statistics_data: Prepared dataframe with serial dependence measures (created by prepare_statistics_serDep() function).
    :param split: How many sections the data sequence has been split into in input datframe.
    :param title: Whether to plot figure with title.
    :param plot_singlesubject: Whether to plot single subject datapoints.
    :param savefig: Whether to save figure.
    
    """
    
    rng = np.random.default_rng()
    
    #Format plot
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot()
    
    xtick = [0.5,0.7]
    
    ax.set_ylim([3, 3])
    ax.set_yticks([-3,-2,-1,0,1,2,3])
    ax.set_xticks(xtick)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0, -1, (np.max(np.unique(modelfree_dependence_statistics_data.split))+0.51), colors='k', linestyles='dotted', lw=0.8)
    ax.set_xlim([0.3,0.9])
    
    #Prepare ticklabels according to split.
    ticklabels = []
    for t in range(len(split)):
        if t+1 != len(split):
            lab = str(split[t]) + "°-" + str(split[t+1]) + "° distance"
        else:
            lab = str(split[t]) + "°-0° distance"
        ticklabels.append(lab)
    
    ax.set_xticklabels(ticklabels)
    ylab = r'$\longleftarrow$' + " repulsion    |    attraction " + r'$\longrightarrow$'
    
    ax.set_ylabel(ylab)
    
    if title:
        fig.suptitle("Model-free measure of stimulus influence on current stimulus report")
    
    splits = np.unique(modelfree_dependence_statistics_data.split).astype(int)
    
    colors_types = ['b', 'y', 'r', 'g']
    markers_types = ['s', 'v', '^', 'o']
    labels_types = ["previous trial's\nGabor orientation", "previous trial's\nresponse orientation",
                    "active template\norientation", "latent template\norientation"]
    
    #Iterate through dependence types:
    for i in np.array([0,2,3]):
        #Prepare variables to store group mean and sem
        tmp_means = np.zeros((len(splits)))
        tmp_sem = np.zeros((len(splits)))
        
        #Select dependence type
        tmp_data = modelfree_dependence_statistics_data[modelfree_dependence_statistics_data['type'] == i]
        
        for k in splits:
            #Select split
            sel = tmp_data[tmp_data['split'] == k]
            
            #Calculate group mean and sem for data section.
            tmp_means[k] = np.mean(sel.SD_measure)
            tmp_sem[k] = stats.sem(sel.SD_measure)
            
            #Store single-subject data for section.
            tmp_ssdata = sel.SD_measure
            
            #Hard-coded: significance stars only at far-range active template dependence.
            if i == 2 and k == 0:
                ax.text(xtick[k]-(0.05), tmp_means[k]-0.15,"***", size=14, weight='semibold')

        #Plot mean dependences.
        ax.errorbar(xtick, tmp_means, tmp_sem, marker=markers_types[i], ms=10, linestyle='None', capsize=3, markerfacecolor=colors_types[i], markeredgewidth=1.2, ecolor='k', markeredgecolor='k', label=labels_types[i], color=colors_types[i])
    
        if plot_singlesubject:
            #Plot single-subject datapoints.
            jitter = rng.integers(low=-5, high=5, size=len(tmp_ssdata))
            jitter = jitter/100
            jitter = np.expand_dims(jitter, 1)
        
            for k in splits:
                x_tmp = np.tile(xtick[k], (len(tmp_ssdata), 1))+jitter
        
                ax.scatter(x=x_tmp, y=tmp_ssdata, color=colors_types[i],marker=markers_types[i], edgecolors=colors_types[i], alpha=0.3, zorder=1, s = 7)
        
    fig.legend(loc=3, prop={'size': 7.8}, frameon = False, bbox_to_anchor=(0.64, 0.68))
    
    if savefig:
        fig.savefig(path_results/"test.png", dpi=900,bbox_inches='tight')

def plot_single_subject_modelfree_dependence(modelfree_dependence_statistics_data, title = False):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Function to plot  model-free dependence measures for multiple dependencies and all subjects.
    
    :param modelfree_dependence_statistics_data: Prepared dataframe with serial dependence measures (created by prepare_statistics_serDep() function).
    :param title: Whether to plot figure with title.
    
    """
    
    subs = np.unique(modelfree_dependence_statistics_data.Sub_ID)
    n_subs = len(subs)
    
    splits = np.unique(modelfree_dependence_statistics_data.split).astype(int)
    
    SD_types = np.arange(len(np.unique(modelfree_dependence_statistics_data.type)))

    figtitles = ['Single-subject measures of previous\nstimulus orientation on response error',
                  "Single-subject measures of previous\nresponse orientation on response error",
                  'Single-subject measures of active\ntemplate orientation on response error',
                  'Single-subject measures of latent\ntemplate orientation on response error']
    
    labels = ['> 45° distance','< 45° distance']
    
    #Variable to return some data as readable dictionary
    keys = ['SD_classic', 'SD_resp', 'SD_actTempl', 'SD_latTempl']
    sign_subjects = {'SD_classic': {}, 'SD_resp': {},
                     'SD_actTempl': {}, 'SD_latTempl': {}}
    
    for i in SD_types:
        #Format plots
        fig, ax = plt.subplots(2)
        xticks = np.arange(n_subs+1)
        gathers = []
        
        for ploti in splits:
            ax[ploti].set_xlim([-1, (n_subs+1)])
            ax[ploti].set_ylim([-6, 6])
            ax[ploti].set_yticks([-6,0, 6])
            ax[ploti].set_xticks(xticks)
            ax[ploti].hlines(0, -1, 23, colors='k', linestyles='dotted', lw=0.8)
            ax[ploti].set_xticklabels([])
            ax[ploti].set_ylabel(r'$\leftarrow$' + " repulsion  |  attraction " + r'$\rightarrow$', fontsize=7.5)
            ax[ploti].yaxis.set_label_coords(-0.045,.5)
            
            ax[ploti].spines['left'].set_color('k')
            ax[ploti].spines['bottom'].set_visible(False)
            ax[ploti].spines['right'].set_visible(False)
            ax[ploti].spines['top'].set_visible(False)
            ax[ploti].tick_params(bottom=False)
    
            gathers.append(np.zeros((n_subs+1, 4)))
            
        if title:
            fig.suptitle(figtitles[i])
                
        markers = ['s', 'o', '^']
        colors = ['b', 'r', 'g']
        meancolors = ['cyan', 'gold']
    
        for subi, sub in enumerate(subs):
            sign_subjects[keys[i]][subi] = []
            
            for k in splits:
                #Select from dataframe data of specific subject, dependence type, and data section.
                sel = modelfree_dependence_statistics_data[(modelfree_dependence_statistics_data['Sub_ID'] == sub) & (modelfree_dependence_statistics_data['type'] == i) & (modelfree_dependence_statistics_data['split'] == k)]
                
                #Extract measure and its significance
                gathers[k][subi, 0] = sel.SD_measure.values[0]
                gathers[k][subi, 2] = sel.significance.values[0]
    
        #After all single-subject measures and significances have been extracted,
        #store mean and standard deviation of those measures.
        for k in splits:
            gathers[k][-1, 0] = np.mean(gathers[k][:, 0])
            gathers[k][-1, 1] = np.std(gathers[k][:, 0])
            
            #The following marks the mean for different plotting:
            gathers[k][-1, 3] = 1
    
        #For each subject, determine the absolute difference in dependence between both sections of data.
        #This is going to be used to sort the data for plotting.
        differ = np.zeros((n_subs+1))
        for d in range(n_subs+1):
            differ[d] = abs(gathers[0][d, 0]-gathers[1][d, 0])
        
        #Sort dependence measures, significances, and mean markers for each data section by differ variable.
        differ, gathers[0][:, 0], gathers[0][:, 1], gathers[0][:, 2], gathers[0][:, 3], gathers[1][:, 0], gathers[1][:, 1], gathers[1][:, 2], gathers[1][:, 3] = zip(*sorted(zip(differ, gathers[0][:, 0], gathers[0][:, 1], gathers[0][:, 2], gathers[0][:, 3], gathers[1][:, 0], gathers[1][:, 1], gathers[1][:, 2], gathers[1][:, 3])))
    
        for k in splits:
            #Plot all individual and the mean dependence measures
            ax[k].errorbar(xticks[:], gathers[k][:, 0], marker=markers[k], linestyle='None',markerfacecolor=colors[k], markeredgewidth=0.5, ecolor='k', markeredgecolor='k',  color=colors[k], ms=7,label = labels[k])
            
            #Plot mean dependence measure again in another color and with error bars
            meanindex = np.where(gathers[k][:, 3] == 1)
            meanindex = meanindex[0]
            
            ax[k].errorbar((xticks[meanindex]), gathers[k][meanindex, 0], gathers[k][meanindex, 1], marker=markers[k], linestyle='None', capsize=3, markerfacecolor=meancolors[k], markeredgewidth=0.5, ecolor='k', markeredgecolor='k',  color=meancolors[k],  ms=6)
            
            #Hard-coded: Only in the case of dependence on active template distance and long range is the mean significant (i.e., only then is a star plotted).
            if i == 2 and k == 0:
                ax[k].text((xticks[meanindex]), gathers[k][meanindex, 0]+0.5, "*", size=10, weight='semibold')
    
            ax[k].legend(loc=3, prop={'size': 8}, frameon = True,framealpha = 1, bbox_to_anchor=(0.72, -.05))
    
        #Plot significance stars
        for k in splits:
            for subi in range(n_subs+1):
                mean_tmp = gathers[k][subi, 0]
                sig_tmp = int(gathers[k][subi, 2])
    
                if subi < len(gathers[k])-1:
                    sign_subjects[keys[i]][subi].append("range: " + str(k) + ", signif: " + str(sig_tmp) + ", dir: " + str(np.sign(mean_tmp)))
    
                if sig_tmp > 0:
                    ax[k].text((xticks[subi]), mean_tmp+0.5, "*", size=10, weight='semibold')
    return sign_subjects