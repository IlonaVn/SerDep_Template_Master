def getAngularDistance(angle1, angle2): #taken from Darinka
    """
    :param angle1: reference angle (i.e., the angle to which the second one is compared to)
    :param angle2: comparison angle
    
    """
    import numpy as np
    #Convert input to array
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)
    
    #Find smallest distance between the two angles
    angularDistance = np.zeros_like(angle1)
    angularDistance[~np.isnan(angle1)] = np.mod(angle1[~np.isnan(angle1)]-angle2[~np.isnan(angle1)]+90, 180)-90
    angularDistance[np.isnan(angle1)] = np.nan
    angularDistance = angularDistance * -1 #clockwise error = +, counter-clockwise error = -
    
    return angularDistance

def direction_extent_of_angle_error_cwccw(firstori,secondori):
    """
    Function to determine the circular difference between two angles.
    
    Output is always the smaller of the two angles, 
    with a positive value denoting secondori as being rotated clockwise with respect to firstori,
    and a negative value denoting secondori as being rotated counterclockwise with respect to firstori.
    
    Can determine any difference, which in some cases is an "error" (as in the memory item reproduction error),
    so function name somewhat of a misnomer.
    
    Assumption: 0° and 180° denote vertical lines and 90° horizontal.
    
    :param firstori: the 'baseline' orientation that the second is compared to
    :param secondori: the 'comparison' orientation that is rotated clockwise or counterclockwise to the 'baseline'
    
    """
    
    temp = firstori-secondori
    diff = 0
    if temp == 90 or temp == -90: #90 and -90 is both just 90° difference
        diff = abs(temp)
        
    if temp > 0 and temp < 90: #if comparison is up to 90° smaller than baseline, it is said to be rotated counterclockwise
        diff = temp*(-1) #e.g.: firstori=107°, secondori=57° --> temp = 50° --> turned to -50° to maintain direction coding
    
    if temp < -90: #if comparison is more than 90° larger than baseline, it is said to be rotated counterclockwise 
        diff = (temp+180)*(-1) #e.g.: firstori=20°, secondori=140° --> temp = -120°
                                #additionally, reported delta angle is always the smaller one. E.g.: if temp = -120°, the correct reported delta angle with coded direction is -40°.
    if temp < 0 and temp > -90: #if comparison is up to 90° larger than baseline, it is said to be rotated clockwise
        diff = abs(temp) #e.g.: firstori=15°, secondori=85° --> temp = -70° --> turned to 70° to maintain direction coding

    if temp > 90: #if comparison is more than 90° smaller than baseline, it is said to be rotated clockwise
        diff = abs(temp-180) #e.g.: firstori=170°, secondori=5° --> temp = 165°
                                #as reported delta angle is always the smaller one, 180° is subtracted in this case. To retain the direction coding, the number is then turned positive: 165° --> 15°
    
    return diff

def direction_extent_of_angle_error_leftright(firstori, secondori):
    """
    Same as direction_extent_of_angle_error_cwccw, but with "left or right in the upper half"
    instead of "clockwise or counterclockwise when looking at the smaller angle"
    
    Used for one pilot.
    """
    temp=firstori-secondori
    diff = 0
    if firstori > 90 and secondori < 90:
        if temp < 90:
            diff=abs(temp+90)
        if temp > 90:
            diff=abs(temp-90)
    if firstori < 90 and secondori > 90:
        if abs(temp) < 90:
            diff=(abs(temp)+90)*(-1)
        if abs(temp) > 90:
            diff= (abs(temp)-90)*(-1)
    if (firstori > 90 and secondori > 90) or (firstori < 90 and secondori < 90):
        if temp < 0:
            diff = abs(temp)
        if temp > 0:
            diff = temp*(-1)
    return diff

def add_memitem_previousrep_angle(df):
    """
    Function to determine the angle between the current trial's Gabor and the previous trial's response (as opposed to previous trial's Gabor)
    """
    import numpy as np
    
    resp_angle = np.zeros((len(df),1))
    resp_angle[0] = np.nan
    for i in range(1,len(df)):
        temp = direction_extent_of_angle_error_cwccw(df.loc[i-1, 'MemItem_Repr'],df.loc[i,'MemItem_angle'])
        resp_angle[i] = temp
    
    return resp_angle

def import_data(path, subject, doublepass = 1, angletraining=False, other_file = None):
    """
    Function that imports subjects data from csv and merges multiple passes.
    
    :param path: Path to subjects csv file
    :param subject: code for subject, to fill in filename
    :param doublepass: if 1, two files are loaded and merged. If 0, only first session is loaded
    :param angletraining: if True, imports not main experiment data, but angletraining data
    """
    import pandas as pd
    import numpy as np
    
    if angletraining:
        file = "rawDat_" + subject + "_Sess_001_angletraining.csv"
        file_name = path + file
        df = pd.read_csv(file_name,delimiter = ',')
        
        idx = 0
        for i in range(len(df)):
            if df.loc[i, 'cwccw_RT'] == 0:
                idx = i
                break
        
        df = df[:idx]
        
    else:
        file = "rawDat_" + subject + "_Sess_001_final.csv"
        file_name = path + file
        df = pd.read_csv(file_name, delimiter = ',')
        df['pass_no'] = np.ones((len(df)))
        
        if doublepass:
            if other_file == None:
                file_2 = "rawDat_" + subject + "_Sess_002_final.csv"
            else:
                file_2 = other_file
            file_name_2 = path + file_2
            df_2 = pd.read_csv(file_name_2, delimiter = ',')
            df_2['pass_no'] = np.repeat(2,len(df))
        
            df = pd.concat((df,df_2)).reset_index(drop=True)

    return df

def fix_template_reproduction_order(df):
    """
    For first 6.5 subjects, template reproduction was cued, but not logged randomly.
    This function is to fix that in those cases.
    
    :param df: trialsequence dataframe for one subject
    
    """
    
    df = df.copy(deep=True)
    
    #make array of True and False according to where template reproduction cues are flipped
    idx = (df['Repr_1_Templ1_cue_first'] == 0) | (df['Repr_2_Templ1_cue_first'] == 0)
    
    #response and reaction time are flipped at the spots where idx is True
    df.loc[idx,['Templ1_Repr_1','Templ2_Repr_1']] = df.loc[idx,['Templ2_Repr_1','Templ1_Repr_1']].values
    df.loc[idx,['Templ1_Repr_2','Templ2_Repr_2']] = df.loc[idx,['Templ2_Repr_2','Templ1_Repr_2']].values
    df.loc[idx,['Templ1_RT_1','Templ2_RT_1']] = df.loc[idx,['Templ2_RT_1','Templ1_RT_1']].values
    df.loc[idx,['Templ1_RT_2','Templ2_RT_2']] = df.loc[idx,['Templ2_RT_2','Templ1_RT_2']].values

    return df

def remove_half_pass(df):
    """
    Function to remove second half of second pass for Subj_021 (who didn't finish the session)
    """
    df = df[:972]
    
    return df

def get_CircError(row, directionality = 1):
    """
    Function to compute circular mean and standard deviation from an array.
    Column from dataframe imported as array using "df.column_name.values"
    
    :param row: array over which to compute mean and standard deviation
    :param directionality: if 1, positive and negative numbers are considered. If 0, only absolute values are considered.
    """

    import numpy as np
    import pycircstat as circ #toolbox for circular statistics
    
    #turn degrees into radians because circ.mean() operates w/ radians
    if directionality == 1:
        error = np.deg2rad(np.asarray(row)) 
        
    elif directionality == 0:
        error = np.deg2rad(np.asarray(np.abs(row)))
    
    #get the circular mean and standard deviation of the imported row, 
    #then turn it back into a degree (bc it's more intuitive that way)
    circ_mean = np.rad2deg(circ.mean(error[~np.isnan(error)])) 
    circ_std = np.rad2deg(circ.std(error[~np.isnan(error)]))
    
    return circ_mean, circ_std

def plot_memitem_reproduction_error(data, split_passes, plot_single_subjects, def_acc = 5, cutoff = 3):
    import pycircstat as circ
    import scipy.stats as stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    subs = np.unique(data.Subject)
    n_subs = len(subs)
    bins = np.linspace(-90, 90, 37)
    colors_passes = ['b','r','g']
    colors_fillbetweens = ['cyan','orange']
    legends_passes = ['Session 1','Session 2']

    if split_passes:
        selTrials = [data[data['pass_no']==1],data[data['pass_no']==2]]
    else:
        selTrials = [data]
    n_cond = len(selTrials)
    
    data_tmp = np.zeros((n_subs, len(selTrials), len(bins)-1))
    pass_means = np.zeros((n_subs,n_cond+1))
    markerlist = []
    
    acc_first = np.zeros((n_subs))
    acc_second = np.zeros((n_subs))
    acc = [acc_first, acc_second]
    
    for subi,sub in enumerate(subs):
        if plot_single_subjects:
            gif = plt.figure()
            axo = gif.add_subplot()
            sisu_title = str(sub)
        markerlist.append(sub)
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
                pass_means[subi,condi] = np.rad2deg(circ.mean(np.deg2rad(abs(tmp))))
                    
            #add to list once for every subject how many percent of trials had errors between -5 and 5° (= their accuracy)
            acc[condi][subi] = len(tmp[(tmp>=-def_acc) & (tmp<=def_acc)])/len(tmp)
        
    mean_tmp = np.zeros((len(selTrials),len(bins)-1))
    sem_tmp = np.zeros((len(selTrials), len(bins)-1))

    tmp_mean = np.nanmean(data_tmp, axis=0)
    tmp_sem = stats.sem(data_tmp, axis = 0, nan_policy = 'omit')
    fig = plt.figure()
    ax = fig.add_subplot()
    for condi in range(len(selTrials)):
        for subi in range(n_subs):
            bin_centers = 0.5*(bins[1:]+bins[:-1])
            ax.plot(bin_centers,data_tmp[subi,condi,:],linestyle = '-',color=colors_passes[condi],alpha=0.1)
        mean_tmp[condi,:] = tmp_mean[condi]
        sem_tmp[condi,:] = tmp_sem[condi]
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        ax.plot(bin_centers,mean_tmp[condi,:],linestyle = '-',color=colors_passes[condi],label=legends_passes[condi])
        plt.fill_between(bin_centers, mean_tmp[condi,:]-sem_tmp[condi,:],mean_tmp[condi,:]+sem_tmp[condi,:], alpha=1, edgecolor=colors_passes[condi], facecolor=colors_fillbetweens[condi])

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
    
    positions = [[.75, .6, .2, .3],[.75, .2, .2, .3]]
    a = fig.add_axes(positions[0])
    b = fig.add_axes(positions[1])
    
    list_axes = [a,b]
    means_tmp = np.zeros((n_subs,2))
    sds_tmp = np.zeros((n_subs, 2))
    
    for subi, sub in enumerate(subs):
        for i in range(2):
            data_tmp = selTrials[i]
            data_tmp = data_tmp[data_tmp['Subject'] == sub]
            
            means_tmp[subi,i] = np.mean(abs(data_tmp.MemItem_Repr_Error))
            sds_tmp[subi,i] = np.rad2deg(circ.std(np.deg2rad(data_tmp.MemItem_Repr_Error)))
            
    output_means = np.zeros((2,2))
    output_sds = np.zeros((2,2))
    
    for i in range(2):
        output_means[i,0] = np.mean(means_tmp[:,i])
        output_means[i,1] = stats.sem(means_tmp[:,i])
        
        output_sds[i,0] = np.mean(sds_tmp[:,i])
        output_sds[i,1] = stats.sem(sds_tmp[:,i])
    
    list_ss_means_sds = [means_tmp,sds_tmp]
    list_means_sds = [output_means, output_sds]
    ylabels = ['Absolute\nresponse error [°]','Imprecision\n[circular SD]']
    ylims = [[4,15],[4,20]]
    yticks = [[5,15],[5,20]]
    hlines_list = [15,20]
    
    for p in range(len(list_axes)):
        rng = np.random.default_rng()
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
        list_axes[p].hlines(hlines_list[p], 0, 1, colors = 'k')
        list_axes[p].text(0.3, hlines_list[p], "***", size=12, weight = 'semibold')
        
        for i in range(2):
            list_axes[p].bar(xticks_means[i], list_means_sds[p][i,0], yerr = list_means_sds[p][i,1],edgecolor='k', color = colors[i], width = 0.3)
            
            jitter = rng.integers(low=-5, high=5, size=len(means_tmp))
            jitter = jitter/100
            jitter = np.expand_dims(jitter, 1)
            
            x_tmp = np.tile(xticks_means[i], (len(means_tmp), 1))+jitter
            
            list_axes[p].errorbar(x_tmp, list_ss_means_sds[p][:,i], marker = 'o', linestyle='None', markeredgecolor = colors_fillbetweens[i], c = 'None', alpha = 0.3)    

    outliers_first= []
    outliers_second = []
    
    overall_mean = np.rad2deg(circ.mean(np.deg2rad(abs(data.MemItem_Repr_Error))))
    if overall_mean > 300:
        overall_mean = overall_mean-360
    
    mean_acc_first = np.nanmean(acc[0][:])
    sem_acc_first = stats.sem(acc[0][:])
    mean_acc_second = np.nanmean(acc[1][:])
    sem_acc_second = stats.sem(acc[1][:])

    for subi, sub in enumerate(subs):
        if acc[0][subi] < mean_acc_first - (cutoff*sem_acc_first) or acc[0][subi] > mean_acc_first + (cutoff*sem_acc_first):
            outliers_first.append(str(sub))
        if acc[1][subi] < mean_acc_second - (cutoff*sem_acc_second) or acc[1][subi] > mean_acc_second + (cutoff*sem_acc_second):
            outliers_second.append(str(sub))
            
    outliers = [outliers_first, outliers_second]
    
    dif = plt.figure()
    axd = dif.add_subplot()
    dottitles = ['first pass','second pass','improvement']
    pass_means[:,n_cond] = pass_means[:,0] - pass_means[:,1]
    pass_means[:,0], markerlist, pass_means[:,1], pass_means[:,n_cond] = zip(*sorted(zip(pass_means[:,0], markerlist, pass_means[:,1], pass_means[:,n_cond])))
    #amps = np.sort(amps)
    x = np.linspace(0,1,len(pass_means))
    xticks = np.linspace(0,1,len(pass_means))
    axd.set_xticks(ticks=xticks)
    axd.set_xticklabels(markerlist, rotation=90)
    
    for condi in range(n_cond+1):
        axd.plot(x,pass_means[:,condi], linestyle = 'None', marker='o',c= colors_passes[condi], label = dottitles[condi])

    axd.set_title("pass comparison of subject-level mean absolute response error")
    dif.legend(loc='center right')
    
    return overall_mean, outliers

def add_memitem_reproduction_error(df, directionality = 1):
    """
    Add column with memory item reproduction errors, plot histogram of that row, 
    and return circular mean and standard deviation of those errors.
    
    :param df: trialsequence dataframe for one subject
    :param qualCheck: if True, reproduction error histogram is plotted
    :param subjectname: code for subject, to add to histogram
    
    """
    import numpy as np
    from MemTempl_analysis_functions_basic import direction_extent_of_angle_error_cwccw, get_CircError
    
    error = np.zeros((len(df)))
    for i in range(len(error)):
        error[i] = direction_extent_of_angle_error_cwccw(df.loc[i,'MemItem_angle'],df.loc[i,'MemItem_Repr'])
    
    df['MemItem_Repr_Error'] = error
    
    av_repr_error, std_repr_error = get_CircError(error, directionality = directionality)
    
    if abs(av_repr_error) > 300:
        av_repr_error = av_repr_error - 360
    
    return df, av_repr_error, std_repr_error

def add_template_reproduction_error_column(df):
    import numpy as np
    from MemTempl_analysis_functions_basic import getAngularDistance

    df.loc[df.miniblock_start == 1, "Templ1_R1_error"] = getAngularDistance(df.loc[df.miniblock_start == 1, "MemTempl1_angle"],df.loc[df.miniblock_start == 1, "Templ1_Repr_1"])
    df.loc[df.miniblock_start == 1, "Templ2_R1_error"] = getAngularDistance(df.loc[df.miniblock_start == 1, "MemTempl2_angle"],df.loc[df.miniblock_start == 1, "Templ2_Repr_1"])
    df.loc[df.miniblock_end == 1, "Templ1_R2_error"] = getAngularDistance(df.loc[df.miniblock_end == 1, "MemTempl1_angle"],df.loc[df.miniblock_end == 1, "Templ1_Repr_2"])
    df.loc[df.miniblock_end == 1, "Templ2_R2_error"] = getAngularDistance(df.loc[df.miniblock_end == 1, "MemTempl2_angle"],df.loc[df.miniblock_end == 1, "Templ2_Repr_2"])
    
    df.loc[((df.miniblock_start == 0) & (df.miniblock_end == 0)), 'Templ1_R1_error'] = np.nan
    df.loc[((df.miniblock_start == 0) & (df.miniblock_end == 0)), 'Templ2_R1_error'] = np.nan
    df.loc[((df.miniblock_start == 0) & (df.miniblock_end == 0)), 'Templ1_R2_error'] = np.nan
    df.loc[((df.miniblock_start == 0) & (df.miniblock_end == 0)), 'Templ2_R2_error'] = np.nan
    
    return df

def distance_cued_template_memitem(df,cwccworleftright,cued, prevorcurr = 'curr'):
    """
    Function to determine the distance and direction 
    between one of the templates (baseline) and the current memory item.
    :param df: trialsequence dataframe for one subject
    :param cwccworleftright: 'cwccw' if the instruction was to determine if the memory item was
                            rotated clockwise or counterclockwise with regard to the smallest angle 
                            between it and the template.
                            'leftright' if the instruction was to determine if the memory item was
                            rotated "to the left" or "to the right" of the template when considering
                            the upper half of the circle.
    :param cued: 'active' if distance between active template and memory item is to be determined.
                'latent' if distance between latent template and memory item is to be determined.
    
    """
    from MemTempl_analysis_functions_basic import direction_extent_of_angle_error_cwccw, direction_extent_of_angle_error_leftright
    import numpy as np
    
    template_memitem_dist = np.zeros((len(df),1))
    
    if prevorcurr == 'curr':
        for i in range(len(df)):
            if cued == 'active':
                if df.loc[i, 'Active'] == 1:
                    template = 'MemTempl1_angle'
                elif df.loc[i, 'Active'] == 2:
                    template = 'MemTempl2_angle'
            elif cued == 'latent':
                if df.loc[i, 'Active'] == 1:
                    template = 'MemTempl2_angle'
                elif df.loc[i, 'Active'] == 2:
                    template = 'MemTempl1_angle'
                    
            if cwccworleftright == 'cwccw':
                temp = direction_extent_of_angle_error_cwccw(df.loc[i, template],df.loc[i,'MemItem_angle'])
                
            elif cwccworleftright == 'leftright':
                temp = direction_extent_of_angle_error_leftright(df.loc[i, template],df.loc[i,'MemItem_angle'])
    
            template_memitem_dist[i] = temp
    
    if prevorcurr == 'prev':
        template_memitem_dist[0] = np.nan
        for i in range(1,len(df)):
            if cued == 'active':
                if df.loc[i-1, 'Active'] == 1:
                    template = 'MemTempl1_angle'
                elif df.loc[i-1, 'Active'] == 2:
                    template = 'MemTempl2_angle'
            elif cued == 'latent':
                if df.loc[i-1, 'Active'] == 1:
                    template = 'MemTempl2_angle'
                elif df.loc[i-1, 'Active'] == 2:
                    template = 'MemTempl1_angle'
                    
            if cwccworleftright == 'cwccw':
                if (i > 0 and i%18 != 0):
                    temp = direction_extent_of_angle_error_cwccw(df.loc[i-1, template],df.loc[i,'MemItem_angle'])
                elif (i%18 == 0 or i <= 0):
                    temp = np.nan
                    
            elif cwccworleftright == 'leftright':
                temp = direction_extent_of_angle_error_leftright(df.loc[i, template],df.loc[i,'MemItem_angle'])
            
            template_memitem_dist[i] = temp
                
    return template_memitem_dist

def add_corrected_resp_error(df, qualCheck = False):
    """
    Function to determine one subject's general response bias depending on absolute stimulus orientation 
    and to subtract this from memitem response error of every trial.
    """
    import math
    import numpy as np
    import pycircstat as circ
    import matplotlib.pyplot as plt
    
    respbiasDependingonOri = df.MemItem_angle.values
    
    responseerrors = df.MemItem_Repr_Error.values
        
    responseerrors[np.isnan(respbiasDependingonOri)] = np.nan
    respbiasDependingonOri[np.isnan(responseerrors)] = np.nan
    
    respbiasDependingonOri = respbiasDependingonOri[~np.isnan(respbiasDependingonOri)]
    responseerrors = responseerrors[~np.isnan(responseerrors)]
    
    #Pad the data (to be able to smooth even those data at the 'edges' of the circle)
    respbiasDependingonOri_padded = respbiasDependingonOri-180
    respbiasDependingonOri_padded = np.append(respbiasDependingonOri_padded, respbiasDependingonOri)
    respbiasDependingonOri_padded = np.append(respbiasDependingonOri_padded, respbiasDependingonOri+180)
    
    responseerrors_padded = np.tile(responseerrors, 3)   
    
    data_smoothed = np.zeros(181) #181

    to_iter = range(0,181)
    index = np.argwhere((to_iter==0) | (to_iter==90) |(to_iter==180))
    to_iter = np.delete(to_iter, index)
    
    for bini in to_iter:
        data_smoothed[bini] = np.rad2deg(circ.mean(np.deg2rad(responseerrors_padded[respbiasDependingonOri_padded == bini])))
    
    for bini in to_iter:
        if (data_smoothed[bini] == 0 or np.isnan(data_smoothed[bini])) and bini != np.max(to_iter):
            data_smoothed[bini] = np.rad2deg(circ.mean(np.deg2rad(np.array([data_smoothed[bini-1], data_smoothed[bini+1]]))))
    
    #data_smoothed = average response error for every absolute orientation
    data_smoothed = np.mod(data_smoothed+90, 180)-90
    
    #smooth this further using something like sum of sine
    if qualCheck:
        fig = plt.figure()
        ax = fig.add_subplot()
        x = np.linspace(0,180,181)
        ax.plot(x, data_smoothed)
    
    resp_error_corrected = np.zeros((len(df)))
    
    for i in range(len(resp_error_corrected)):
        currerr = df.loc[i, 'MemItem_Repr_Error']
        currerr = currerr.astype(int)
        if not math.isnan(currerr):
            currori = df.loc[i, 'MemItem_angle']
            currori = currori.astype(int)
            
            currmodel = data_smoothed[currori]
            
            resp_error_corrected[i] = currerr - currmodel
        else:
            resp_error_corrected[i] = np.nan
    
    return resp_error_corrected

def determine_switch_nonswitch(df):
    import numpy as np
    switch_nonswitch = np.zeros((len(df)))
    for i in range(len(df)):
        if i%18 == 0:
            switch_nonswitch[i] = np.nan
        else:
            if df.loc[i, 'Active'] != df.loc[i-1, 'Active']:
                switch_nonswitch[i] = 1
    
    return switch_nonswitch

def cwccw_performance_column(df):
    """
    Function to determine if participant made correct ("1") or incorrect ("0") cwccw judgments
    by comparing answers with predetermined correct answers from trialsequence (from determine_correct_answers_cwccw()).
    Returns column denoting if response was correct for any given trial
    
    :param df: complete or part of trialsequence dataframe for one subject
    
    """
    import numpy as np
    
    cwccw_correct = np.zeros((len(df)))
    
    for i in range(len(df)):
        if df.loc[i,'correct_cwccw'] == df.loc[i,'cwccw_resp'] or df.loc[i, 'correct_cwccw'] == 2:
            cwccw_correct[i] = 1
    
    return cwccw_correct

def add_distortion_condition(df):
    import numpy as np
    
    distortion = np.zeros((len(df)))
    for i in range(len(distortion)):
        if i%18 == 0:
            distortion[i] = np.nan
        else:
            if abs(df.loc[i-1,"distance_active_item"]) >= 45:
                distortion[i] = 1
            elif abs(df.loc[i-1,"distance_active_item"]) < 45:
                distortion[i] = 0
            
    return distortion

def get_memitem_RT(df):
    import numpy as np
    
    RT_values = df.MemItem_RT
    final_RTs = np.zeros((len(RT_values)))
    for i in range(len(RT_values)):
        list_of_strings = RT_values[i][1:-1].split(',')
        array_of_floats = np.array(list_of_strings, dtype='float')
        final_RTs[i] = array_of_floats[-1]
    
    return final_RTs

def find_trial_length(df):
    """
    Individual trial length for each trial, depending on constant stimuli (4.4 seconds)
    and response times.
    """
    import numpy as np
    final_trial_length = np.zeros((len(df)))
    for i in range(len(final_trial_length)):
        final_trial_length[i] = 4.4 + df.loc[i, "cwccw_rt"] + df.loc[i, "clean_MemItem_RT"] #4.4 for fixation crosses, cues, and stimuli
    
    return final_trial_length

def find_reproduction_correlation(array1, array2, circ1 = True, circ2 = True):
    """
    Function to find correlation coefficient between two variables.
    :param circ1: whether array1 is circular
    :param circ2: whether array2 is circular
    """
    import numpy as np
    import pycircstat as circ
    
    if circ1:
        array1 = np.deg2rad(array1)
    if circ2:
        array2 = np.deg2rad(array2)
    
    if circ1 and circ2:
        #function for correlation between two circular variables
        corr = circ.corrcc(array1, array2)
    
    if circ1 == False and circ2 == False:
        corr = np.corrcoef(array1, array2)
        
    return corr

def determine_template_reproduction_correlation(data):
    import numpy as np
    from MemTempl_analysis_functions_basic import find_reproduction_correlation
    
    subs = np.unique(data.Subject)
    
    corr_dict = {}
    for subi, sub in enumerate(subs):
        tmp = np.zeros((3))
        
        #determine correlation between template ori and first resp, template ori and second resp, and first and second resp
        ori1 = data.loc[((data['Subject'] == sub) & (data['miniblock_start'] == 1)), 'MemTempl1_angle']
        ori2 = data.loc[((data['Subject'] == sub) & (data['miniblock_start'] == 1)), 'MemTempl2_angle']
        ori = np.concatenate((ori1, ori2))
        
        resp11 = data.loc[((data['Subject'] == sub) & (data['miniblock_start'] == 1)), 'Templ1_Repr_1']
        resp12 = data.loc[((data['Subject'] == sub) & (data['miniblock_start'] == 1)), 'Templ2_Repr_1']
        resp1 = np.concatenate((resp11,resp12))
        
        resp21 = data.loc[((data['Subject'] == sub) & (data['miniblock_end'] == 1)), 'Templ1_Repr_2']
        resp22 = data.loc[((data['Subject'] == sub) & (data['miniblock_end'] == 1)), 'Templ2_Repr_2']
        resp2 = np.concatenate((resp21,resp22))
        
        tmp[0] = find_reproduction_correlation(ori, resp1)
        tmp[1] = find_reproduction_correlation(ori, resp2)
        tmp[2] = find_reproduction_correlation(resp1, resp2)
        
        corr_dict[sub] = tmp
    
    return corr_dict

def determine_memitem_reproduction_correlation(data):
    import numpy as np
    from MemTempl_analysis_functions_basic import find_reproduction_correlation
    
    subs = np.unique(data.Subject)

    corr_dict = {}
    for subi, sub in enumerate(subs):
        ori = data.loc[data['Subject'] == sub, 'MemItem_angle']
        resp = data.loc[data['Subject'] == sub, 'MemItem_Repr']
        corr_dict[sub] = find_reproduction_correlation(ori, resp)
    
    return corr_dict

def determine_cwccw_perccorrect(data):
    import numpy as np
    
    subs = np.unique(data.Subject)
    perf_dict = {}
    for subi, sub in enumerate(subs):
        sel = data[data['Subject'] == sub]
        perf_dict[sub] = np.count_nonzero(sel['performance_cwccw'] == 1)/len(sel)
    
    allvalues = list(perf_dict.values())
    
    mean_perf = np.mean(np.array((allvalues)))
    sd_perf = np.std(np.array((allvalues)))
    
    perf_dict['mean'] = mean_perf
    perf_dict['sd'] = sd_perf
    
    return perf_dict

def cwccw_response_ratio_distance(df,active_latent, response = 'ccw', binsize = 10, rangelim = 90):
    """
    Function to determine the ratio of ccw or cw responses to cwccw task for each bin of distance of
    either the active or latent template to memory item.
    
    Used to then plot these ratios with plot_active_latent_together().
    
    :param df: complete or part of trialsequence dataframe for one subject
    :param active_latent: determine ratios according to distance between active or latent template and memory item
    :param response: give ratio of ccw or cw responses
    :param binsize: how big are the bins into which distances are rounded
    
    """
    import numpy as np
    import pandas as pd
    
    #determine number and size of bins according to input specifications
    binnumber = int(rangelim*2)/binsize
    bins = np.linspace(-rangelim,rangelim-10,int(binnumber))
    
    if response == 'ccw':
        r = 0
    elif response == 'cw':
        r = 1
        
    #determine distances between either active or latent template and current item in every trial
    template_memitem_distance = distance_cued_template_memitem(df,'cwccw',active_latent)
    template_memitem_distance = template_memitem_distance.reshape(len(template_memitem_distance),1)
    
    #round each trial's distance into respective bin
    for s in bins:
        for l in range(0,len(template_memitem_distance)):
            if template_memitem_distance[l] > s and template_memitem_distance[l] < s+binsize:
                template_memitem_distance[l] = s
    
    #prepare array to input the ratio for each bin
    ratios = np.zeros((len(bins)))
    
    #pull responses to cwccw task from dataframe
    answers = np.array(df['cwccw_resp']).reshape(len(df['cwccw_resp']),1)
    
    #put distances and responses into one dataframe
    tog = np.concatenate((answers,template_memitem_distance),axis=1)
    answers_df = pd.DataFrame(tog, columns=['cwccw_resp','template_memitem_dist'])
    
    #iterate through bins and their indexes
    for x, y in zip(bins, range(len(ratios))):
        #for every bin, pull those trials from new dataframe that fall into that bin
        sel = answers_df[answers_df['template_memitem_dist']==x]
        
        if len(sel) != 0:
            #if there are trials that fall into that bin, determine ratio of cases in which
            #the response was counterclockwise (="0")
            ratio = np.count_nonzero(sel['cwccw_resp'] == r)/len(sel)
        else:
            #if there are no trials that fall into that bin, let ratio be 0 (doesn't happen in full experiment)
            ratio = 0
            
        ratios[y] = ratio
        
    return ratios

def pulldata(data, SD_type, subject = None, n_back = 4):
    """
    Pull right column of data according to SD_type aimed at.
    :param data: cleaned data (no outliers, demeaned)
    """      
    
    import numpy as np
    from MemTempl_analysis_functions_basic import add_control_deltangle
    
    rem_bias = 0
    if SD_type == "SD_classic": #x = angle between current Gabor and previous Gabor, y = difference reproduction of current Gabor and actual current Gabor orientation
        data.loc[data.miniblock_start == 1, "MemItem_deltangle"] = np.nan
        x = data.MemItem_deltangle.values
        title = "SD between current and previous MemItem"
        column = 'MemItem_deltangle'
        xlabel = "Distance of current to previous trial's Gabor [°]"
    
    elif SD_type == 'SD_resp': #x = angle between current Gabor and previous trial's response
        data.loc[data.miniblock_start == 1, "distance_previous_resp"] = np.nan
        rem_bias = data.MemItem_angle.values
        x = data.distance_previous_resp.values
        title = "Effect of distance between previous and current Gabor orientation\non Gabor reproduction response bias"
        column = 'distance_previous_resp'
        xlabel = "Distance of current Gabor to previous trial's response"
    
    elif SD_type == 'SD_currori': #x = current trial's absolute Gabor orientation
        x = data.MemItem_angle.values
        rem_bias = data.MemItem_angle.values
        title = 'Effect of current absolute Gabor orientation on response bias'
        column = ''
        xlabel = ''

        
    elif SD_type == "SD_actTempl": #x = angle between current Gabor and active template orientation, y = difference reproduction of current Gabor and actual current Gabor orientation
        # idx = data.index[data['miniblock_end']==1].tolist()
        # idx = np.array(idx)
        # idx_long = np.zeros((9*72))
        # m = 0
        # for count,i in enumerate(idx):
        #     tmp = np.arange(i-9,i)
        #     idx_long[count:int(count+9)] = tmp
        #     m += 9
        # idx_long.tolist()
        # data = data.iloc[idx_long]
        x = data.distance_active_item.values
        title = "Effect of distance between currently active template orientation\non Gabor reproduction response bias"
        column = 'distance_active_item'
        xlabel = "Distance of current Gabor to active template [°]"
        #y = data.Resp_error_demeaned.values
    
    elif SD_type == "SD_latTempl": #x = angle between current Gabor and latent template orientation, y = difference reproduction of current Gabor and actual current Gabor orientation
        x = data.distance_latent_item.values
        title = "Effect of distance between currently latent template orientation\non Gabor reproduction response bias"
        column = 'distance_latent_item'
        xlabel = "Distance of current Gabor to latent template [°]"
        
    elif SD_type == 'SD_prevactTempl':
        x = data.distance_prevactive_item.values
        title = "SD between current MemItem and previously active template"
        column = 'distance_prevactive_item'
        xlabel = ''
    
    elif SD_type == 'SD_prevlatTempl':
        x = data.distance_prevlatent_item.values
        title = "SD between current MemItem and previously latent template"
        column = 'distance_prevlatent_item'
        xlabel = ''
    
    elif SD_type == 'SD_control':
        #data = data[data['Subject'] == subject]
        data['Control_ori'] = add_control_deltangle(data, n_back = n_back, subject = subject)
        
        index = data.index
        cond = data["miniblock_start"] == 1
        idx = index[cond]
        
        idx_list = idx.tolist()
        for i in range(len(idx_list)):
            data.loc[idx_list[i]:idx_list[i]+(n_back-1), 'Control_ori' ] = np.nan
        
        x = data.Control_ori.values
        title = 'Control - SD between current MemItem and ' + str(n_back) + ' back trial stimulus'
        xlabel = ''
    
    return x, title, column, xlabel

def add_control_deltangle(df, subject, n_back = 4):
    import numpy as np

    control_deltangles = np.zeros((len(df),1))
    for i in range(n_back,len(df)):
        temp = direction_extent_of_angle_error_cwccw(df.loc[i-n_back, 'MemItem_angle'],df.loc[i,'MemItem_angle'])
        control_deltangles[i] = temp
    #control_deltangles = np.pad(control_deltangles, (n_back, 0), 'constant')
    #control_deltangles[:n_back] = np.nan
    
    return control_deltangles

def return_modelfreemeasure_nonsmoothed(data_clean, SD_type, split = [90], meanormedian = 'median', degreeorradian= 'degree'):
    """
    Function to return model-free measure of dependence of response error on different independent variables.
    
    Modeled after Gallagher 2022: there, median response errors between 0° and 45° and between 45° and 90° were subtracted
    from those between 0° and -45° and between -45° and -90°, respectively.
    
    The result is the model-free measure of dependence used in Gallagher 2022: Stimulus uncertainty predicts serial dependence in orientation judgements.
    
    :param data_clean: Cleaned data for all subjects (cleaned by remove_outliers_SD() function)
    :param SD_type: The independent variable whose impact on response error is wanted.
    :param split: From a scale from 0° to 90°, where cutoffs for multiple separate dependence measures at specific distances should be computed.
                    90 = default, this return one overall dependence measure.
    :param meanormedian: Whether the measure should reflect the difference in means or medians between sides. Gallagher 2022 used medians.
    :param degreeorradian: Return dependence measure in degrees or radians.
    
    """
    import numpy as np
    import pycircstat as circ
    from MemTempl_analysis_functions_basic import pulldata
    
    subs = np.unique(data_clean.Subject)
    
    #Prepare variable to collect each subject's measure of 
    #serial dependence for each range of distances specified
    modfree_params = np.zeros((len(subs),len(split)))
    
    for subi, sub in enumerate(subs):
        #Pull sequence of independent variable specified
        x, titlename,_,_ = pulldata(data_clean[data_clean['Subject'] == sub], SD_type, n_back = 4, subject = '')

        #Get respective cleaned response error sequence
        if SD_type == 'SD_resp' or SD_type == 'SD_currori':
            y = data_clean[data_clean['Subject'] == sub].Corr_resp_error_demeaned.values
        else:
            y = data_clean[data_clean['Subject'] == sub].Resp_error_demeaned.values
        
        y[np.isnan(x)] = np.nan
        x[np.isnan(y)] = np.nan
        
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        #Variables to separately store data from each side of the plot
        #length of these variables will correspond to number of sections to receive a measure of dependence.
        splitdataccw = []
        splitdatacw = []
        
        ##According to number and placement of cutoffs for multiple measures of dependence,
        ##split response error datapoints
        #of the counterclockwise side
        if len(split) == 1:
            splitdataccw.append(y[(x > -split[0]) & (x < 0)])
        elif len(split) == 2:
            splitdataccw.append(y[(x > -split[0]) & (x < -split[1])])
            splitdataccw.append(y[(x > -split[1]) & (x < 0)])
        elif len(split) == 3:
            splitdataccw.append(y[(x > -split[0]) & (x < -split[1])])
            splitdataccw.append(y[(x > -split[1]) & (x < -split[2])])
            splitdataccw.append(y[(x > -split[2]) & (x < 0)])
        
        #of the clockwise side
        if len(split) == 1:
            splitdatacw.append(y[(x < split[0]) & (x > 0)])
        elif len(split) == 2:
            splitdatacw.append(y[(x < split[0]) & (x > split[1])])
            splitdatacw.append(y[(x < split[1]) & (x > 0)])
        elif len(split) == 3:
            splitdatacw.append(y[(x < split[0]) & (x > split[1])])
            splitdatacw.append(y[(x < split[1]) & (x > split[2])])
            splitdatacw.append(y[(x < split[2]) & (x > 0)])
        
        #For each section, determine dependence measure by subtracting
        #one side's mean with the other
        for k in range(len(split)):
            ccwsec = splitdataccw[k]
            cwsec = splitdatacw[k]

            if meanormedian == 'median': #as in Gallagher 2022
                ccw_median = np.around(np.median(ccwsec), decimals = 4)
                cw_median = np.around(np.median(cwsec), decimals = 4)
                
                if degreeorradian == 'radian':
                    modfree_params[subi,k] = np.deg2rad(ccw_median-cw_median)
                elif degreeorradian == 'degree':
                    modfree_params[subi,k] = ccw_median-cw_median
            
            elif meanormedian == 'mean':
                ccw_mean = np.rad2deg(circ.mean(np.deg2rad(ccwsec)))
                if ccw_mean > 300:
                    ccw_mean = ccw_mean-360
                    
                cw_mean = np.rad2deg(circ.mean(np.deg2rad(cwsec)))
                if cw_mean > 300:
                    cw_mean = cw_mean-360
            
                if degreeorradian == 'radian':
                    modfree_params[subi,k] = np.deg2rad(ccw_mean-cw_mean)
                elif degreeorradian == 'degree':
                    modfree_params[subi,k] = ccw_mean-cw_mean
                    
    return modfree_params