# -*- coding: utf-8 -*-

#Purpose: Create randomized and counter-balanced trial sequence
#Author: Darinka Truebutschek/Ilona Vieten
#Date created: 25/05/2021
#Date last modified: 21/04/2022
#Python version: 3.6.13

### Import needed tools ###
import numpy as np
import pandas as pd

from numpy.random import shuffle
from numpy import genfromtxt

from MemTempl_config import angleDef, n_angle_bins, n_template1_bins, n_template2_bins, time_iti, n_miniblocks_before_short_break, n_miniblocks_before_long_break, n_miniblocks_per_pass, path_colors, probe, training, test_button_assignment_shuffle
from MemTempl_Taskfunctions import direction_extent_of_angle_error_cwccw,determine_correct_answers_cwccw

#%%
def SerDep_prepTrials():
    """
    Function to generate trial sequence for main task.
    
    I: Creation of unique sequence of combinations of Template 1/Template 2/Gabor orientation bin indices.
    II: Creation of pseudo-randomized sequence of 1s and 2s to denote number of active template in any trial.
    III: Creation of markers for break/pause trials.
    IV: Random determination of exact template orientations based on bin order from I.
    V: Random determination of exact Gabor orientations based on bin order from I (two methods).
    VI: Assign random color sequence to templates.
    VII: Randomly assign cue presentation for judgment task.
    VIII: Marking trials in which both template orientations are on either side of Gabor orientations (in which "Templates frame the Gabor").
    (v): Add angular distance between subsequent Gabors in the case of independently determined absolute orientations.
    IX: Add ITIs.
    X: Add markers for start and end trials of miniblocks.
    XI: Pseudo-randomize performance of judgment task.
    XII: Randomize sequence of template reproductions.
    XIII: Number miniblocks sequentially.
    XIV: Add correct responses to judgment task.
    XV: Add empty rows to be filled during experiment.
    """    
    
    """
    I
    """
    #Create number of conditions
    n_conditions = n_angle_bins*n_template1_bins*n_template2_bins
    
    ##Create basic condition units (basic unit = miniblock)
    #number of trials needed in a single miniblock
    basicUnit_len = n_angle_bins 
    #flat array of unshuffled memory item conditions
    basicUnit = np.array(np.arange(basicUnit_len)) 
    
    ##Create template arrays in a way that when concatenated/stacked, every unique combination is present 1 time
    #Create array of template 1
    #flat array of unshuffled conditions of memory template 1
    order_templ1 = np.repeat(np.arange(n_template1_bins),n_template2_bins) 
    
    #Create array of template 2
    #flat array of unshuffled conditions of memory template 2
    order_templ2 = np.tile(np.arange(n_template2_bins),n_template1_bins)
    
    #stack arrays of templates 1 and 2
    both_blocks = np.column_stack((order_templ1,order_templ2))
    
    #index ordered template conditions, add to other rows
    miniblock_index = np.arange(n_template1_bins*n_template2_bins).reshape((n_template1_bins*n_template2_bins),1)
    all_blocks = np.concatenate((both_blocks,miniblock_index), axis=1)
    
    #shuffle the rows
    shuffle(all_blocks)
    
    ##at this point, all block conditions are present once, indexed in order, and then randomized --> all this has to happen once for every participant
    
    ##Create array with all trials and task conditions
    
    #Prepare one row of zeros for the memory items
    full_memitem = np.zeros((n_conditions,1))
    #Prepare two rows of zeros for the memory templates
    full_templts = np.zeros((n_conditions,2))
    #Prepare one row of zeros for the template indices
    full_templt_indexes = np.zeros((n_conditions,1))

    #Block counter
    m = 0
    
    #Turn flat array of memory item conditions into a row (still unshuffled) of same shape as subpart of full_memitem that is to be filled
    memitems = basicUnit.reshape(basicUnit_len,1)
    
    #Iterate through the uniquely shuffled block conditions to fill out conditions within blocks
    for cond in all_blocks: 
        
        #Shuffle ordered array at first go; re-shuffle shuffled array at every following iteration
        shuffle(memitems) 

        #Counter moves up at many spots as there are memory item conditions at every iteration, shuffled memory item conditions are placed into the space
        full_memitem[m:m+n_angle_bins,:] = memitems 
                                                    
        #Counter moves up at many spots as there are memory item conditions at every iteration (= length of miniblock),
        #template 1 spot of every condition that is iterated through is repeated for the length of the miniblock.
        full_templts[m:m+n_angle_bins,0] = np.array(np.tile(cond[0],n_angle_bins)) 
                                                                                    
        #Same procedure for template 2
        full_templts[m:m+n_angle_bins,1] = np.array(np.tile(cond[1],n_angle_bins)) 
        
        #Same procedure for template condition indices
        full_templt_indexes[m:m+n_angle_bins,0] = np.array(np.tile(cond[2],n_angle_bins))
        
        #Move counter to next miniblock
        m += n_angle_bins

    #Create finished array displaying all conditions in the order that will be presented to subject
    all_conditions = np.concatenate((full_templts,full_memitem,full_templt_indexes),axis=1)
                                                                                            
    #Convert into a pandas data frame and add column names
    df_trialSeq = pd.DataFrame(all_conditions, columns=['MemTempl1_bin','MemTempl2_bin','MemItem_bin','Templ_index'])

    """
    II
    """
    
    ##Add pseudo-randomized row denoting whether template 1 or template 2 will be active in judgment task
    n_condit_half = int((n_angle_bins/3)/2) # half of a third of a miniblock (3 trials)
    
    templ1_act = np.ones((n_condit_half,1)) # make string of 1s
    templ2_act = np.full((n_condit_half,1),2) # make string of 2s
    
    third_of_miniblock = n_angle_bins/3 # determine length of both of those combined (6 trials)
    
    act_lat = np.zeros((n_conditions)).reshape(n_conditions,1) # prepare row of zeros
    
    #Counter for iteration
    m = 0

    for i in range(int(len(act_lat)/third_of_miniblock)): # repeat the following number of conditions/6 times
        together = np.concatenate((templ1_act,templ2_act)) # bring together the strings of 1s and 2s
        shuffle(together) # shuffle the 6 numbers
        
        act_lat[int(m):int(m+third_of_miniblock)] = together # paste them into the prepared string of zeros
        m += third_of_miniblock # count up 6 spots
    
    #Add shuffled 1s and 2s to dataframe
    df_trialSeq.insert(4, 'Active', act_lat)
    
    """
    III
    """
    
    #add row denoting whether there will be a long break after any trial
    pause_trial = np.zeros((n_conditions,1))
    n = 0
    
    for i in range(len(pause_trial)):
        if i%(n_angle_bins*n_miniblocks_before_long_break) == 0 and i != 0:
            pause_trial[n-1] = 1
        n += 1
        
    df_trialSeq.insert(5, 'Long_Break', pause_trial)
    
    #add row denoting whether there will be a short break after any trial
    break_trial = np.zeros((n_conditions,1))
    n = 0

    for i in range(len(break_trial)): # iterate through row of zeros
        if i%(n_angle_bins*n_miniblocks_before_short_break) == 0 and i != 0 and df_trialSeq['Long_Break'][n-1] != 1: # if a trial is reached whose index matches the number of blocks that
                                                                    # should pass before a break (specified in "configuration") and there is no long pause at that point,
            break_trial[n-1] = 1                                    # the 0 at the trial before is turned into a 1
        n += 1
    
    df_trialSeq.insert(5, 'Short_Break', break_trial) # add this list to dataframe
    
    """
    IV
    """
    
    ##Fill in actual template angles (in degrees, randomized within each bin)
    
    #variable to collect exact template orientations so they will not be repeated.
    identities = []
    
    ##Template 1
    #Until now, there've only been indices --> now turn into actual angle bins
    templ1_angleBins = np.arange(0,180,180/n_template1_bins) 
    templ1_angleBins = templ1_angleBins+5 # add 5 to all of them to get them away from 0/180 degrees
    
    #Pull unique index sequence from dataframe
    templ1_angleOrient = np.copy(df_trialSeq['MemTempl1_bin'].values)
    templ1_angleOrient.astype(int)
    
    #Transform index sequence to bin sequence
    for index, v in np.ndenumerate(templ1_angleOrient):
        templ1_angleOrient[index] = templ1_angleBins[int(v)]   
    
    #Iterate through start values of bins (len 6)
    for bini in range(len(templ1_angleBins)):
        #Iterate through unique sequence of bin start values (len 648)
        for triali in range(len(templ1_angleOrient)):
            #This procedure for every bin but the last
            if bini < len(templ1_angleBins)-1:
                #for each point in the sequence that falls within the current bin and is at the start of a miniblock, a random number
                #between the starts of the current and the next bin is inserted, if the orientation has not been used before
                if (templ1_angleOrient[triali] == templ1_angleBins[bini]) and (triali%n_angle_bins == 0):
                    #counter for attempts to find unique orientation
                    m = 0
                    while m == 0 or (templ1_angleOrient[triali] in identities):
                        templ1_angleOrient[triali] = np.random.randint(templ1_angleBins[bini],templ1_angleBins[bini+1])
                        m += 1
                    identities.append(templ1_angleOrient[triali])
                
                #template orientation is only newly determined for the first trial of every miniblock, the other trials receive a copy of the same
                elif (triali%n_angle_bins != 0):
                    templ1_angleOrient[triali] = templ1_angleOrient[triali-1]
            
            #Another procedure is used for the last bin
            else:
                #for each point in the sequence that falls within the current (last) bin, a random number is inserted 
                #that is between the start and end number of the bin, if this number has not been used before and is no greater than 179.
                if (templ1_angleOrient[triali] == templ1_angleBins[bini]) and (triali%n_angle_bins == 0):
                    #counter for attempts to find unique orientation
                    m = 0
                    while m == 0 or (templ1_angleOrient[triali] in identities) or templ1_angleOrient[triali] >= 180:
                        templ1_angleOrient[triali] = np.random.randint(templ1_angleBins[bini], templ1_angleBins[bini]+(180/n_template1_bins))
                        m += 1
                    identities.append(templ1_angleOrient[triali])
                    
                #template orientation is only newly determined for the first trial of every miniblock, the other trials receive a copy of the same
                elif (triali%n_angle_bins != 0):
                    templ1_angleOrient[triali] = templ1_angleOrient[triali-1]
    
    #Add new randomized angles within their bins assigned via sequence to dataframe, to the right of the bins
    df_trialSeq.insert(3, 'MemTempl1_angle', templ1_angleOrient)
    
    ##Template 2 (same procedure as for Template 1)
    #Until now, there've only been indices --> now turn into actual angle bins
    templ2_angleBins = np.arange(0,180,180/n_template2_bins)
    templ2_angleBins = templ2_angleBins+5
    
    #Pull unique index sequence from dataframe
    templ2_angleOrient = np.copy(df_trialSeq['MemTempl2_bin'].values)
    templ2_angleOrient.astype(int)
    
    #Transform index sequence to bin sequence
    for index, v in np.ndenumerate(templ2_angleOrient):
        templ2_angleOrient[index] = templ2_angleBins[int(v)]   
    
    #Iterate through start values of bins (len 6)
    for bini in range(len(templ2_angleBins)):
    #Iterate through unique sequence of bin start values (len 648)
        for triali in range(len(templ2_angleOrient)):
            #This procedure for every bin but the last
            if bini < len(templ2_angleBins)-1:
                #for each point in the sequence that falls within the current bin and is at the start of a miniblock, a random number
                #between the starts of the current and the next bin is inserted, if the orientation has not been used before 
                #or the difference between the currently generated template and its corresponding template 1 partner is less than 10
                if (templ2_angleOrient[triali] == templ2_angleBins[bini]) and (triali%n_angle_bins == 0 or triali == 0): 
                    #counter for attempts to find unique orientation
                    m = 0
                    while m == 0 or (templ2_angleOrient[triali] in identities) or abs(direction_extent_of_angle_error_cwccw(df_trialSeq['MemTempl1_angle'][triali],templ2_angleOrient[triali])) < 10:
                        templ2_angleOrient[triali] = np.random.randint(templ2_angleBins[bini],templ2_angleBins[bini+1]) 
                        m += 1
                    identities.append(templ2_angleOrient[triali])
                    
                #template orientation is only newly determined for the first trial of every miniblock, the other trials receive a copy of the same
                elif (triali%n_angle_bins != 0):
                    templ2_angleOrient[triali] = templ2_angleOrient[triali-1]
            
            #Another procedure is used for the last bin
            else:
                #for each point in the sequence that falls within the current (last) bin, a random number is inserted 
                #that is between the start and end number of the bin, if this number has not been used before and is no greater than 179 and less than 10° apart from corresponding template 1 orientation.
                if (templ2_angleOrient[triali] == templ2_angleBins[bini]) and (triali%n_angle_bins == 0 or triali == 0):
                    #counter for attempts to find unique orientation
                    m = 0
                    while m == 0 or (templ2_angleOrient[triali] in identities) or abs(direction_extent_of_angle_error_cwccw(df_trialSeq['MemTempl1_angle'][triali],templ2_angleOrient[triali])) < 10 or templ2_angleOrient[triali] >= 180:        
                        templ2_angleOrient[triali] = np.random.randint(templ2_angleBins[bini], templ2_angleBins[bini]+(180/n_template2_bins)) # at this spot in the sequence, the bin start value is replaced with a random value between this (last) bin's start value and the last possible value (in this case, last start value + 30 --> between 155 and 185)
                        m += 1 
                    identities.append(templ2_angleOrient[triali])
                #template orientation is only newly determined for the first trial of every miniblock, the other trials receive a copy of the same
                elif (triali%n_angle_bins != 0):
                    templ2_angleOrient[triali] = templ2_angleOrient[triali-1]
    
    #Add new randomized angles within their bins assigned via sequence to dataframe, to the right of the bins
    df_trialSeq.insert(4, 'MemTempl2_angle', templ2_angleOrient)
    
    """
    V
    """
    
    ##Fill in actual memory angle 
    #Method I: In degrees, randomized within each bin analogously to Template orientations
    
    if angleDef == 'cb_angles':
        mem_angleBins = np.arange(0,180,180/n_angle_bins)
        mem_angleBins = mem_angleBins+5
    
        mem_angleOrient = np.copy(df_trialSeq['MemItem_bin'].values)
        mem_angleOrient.astype(int)
        
        for index, v in np.ndenumerate(mem_angleOrient):
            mem_angleOrient[index] = mem_angleBins[int(v)]
        
        for bini in range(len(mem_angleBins)):
            for triali in range(len(mem_angleOrient)):
                if n_angle_bins == 18:
                    if bini < len(mem_angleBins)-1:
                        if (mem_angleOrient[triali] == mem_angleBins[bini]):
                            mem_angleOrient[triali] = np.random.randint(mem_angleBins[bini],mem_angleBins[bini+1])
                    else:
                        if (mem_angleOrient[triali] == mem_angleBins[bini]):
                            mem_angleOrient[triali] = np.random.randint(mem_angleBins[bini], mem_angleBins[bini]+(180/n_angle_bins))
                            #If the randomly determined value exceeds 180, the orientation is instead placed between 0 and 5
                            if mem_angleOrient[triali] >= 180:
                                mem_angleOrient[triali] = mem_angleOrient[triali]-180
        
        #Add new randomized angles within their bins assigned via sequence to dataframe, to the right of the bins    
        df_trialSeq.insert(5, 'MemItem_angle', mem_angleOrient)                        
    
    #Method II: In degrees, drawn from randomized and counterbalanced distances between sequential memory items

    elif angleDef == "cb_deltaAngles":
        
        #First, determine random sequence of distance from previous Gabor analogously to previous section.
        mem_deltangleBins = np.linspace(-90, 90, n_angle_bins+1)

        mem_deltangles = np.copy(df_trialSeq['MemItem_bin'].values)
        mem_deltangles.astype(int)

        for index, v in np.ndenumerate(mem_deltangles):
            mem_deltangles[index] = mem_deltangleBins[int(v)]
        
        for bini in range(len(mem_deltangleBins)-1):
            for triali in range(len(mem_deltangles)):
                if bini < len(mem_deltangleBins)-2:
                    if mem_deltangles[triali] == mem_deltangleBins[bini]:
                        mem_deltangles[triali] = np.random.randint(mem_deltangleBins[bini],mem_deltangleBins[bini+1])
                else:
                    if mem_deltangles[triali] == mem_deltangleBins[bini]:
                            mem_deltangles[triali] = np.random.randint(mem_deltangleBins[bini], mem_deltangleBins[bini]+(180/n_angle_bins))
                            if mem_deltangles[triali] > 90:
                                print('does this actually happen')
                                mem_deltangles[triali] = mem_deltangles[triali]-180
        
        #Add new randomized angle differences within their bins assigned via sequence to dataframe, to the right of the bins                   
        df_trialSeq.insert(5, 'MemItem_deltangle', mem_deltangles)
        
        #Next, determine absolute orientations of Gabors. Randomly draw initial Gabor orientation 
        #for each block, then determine every following orientation from angle differences from last section.
        mem_angles = np.zeros((len(df_trialSeq)))
        for triali in range(len(df_trialSeq)):
            if triali == 0 or triali%n_angle_bins == 0:
                mem_angles[triali] = np.random.randint(1,180) # low inclusive, high exclusive --> no 0 or 180
            
            else:
                mem_angles[triali] = mem_angles[triali-1] + df_trialSeq['MemItem_deltangle'][triali]
                while mem_angles[triali] < 0 or mem_angles[triali] > 180:
                    if mem_angles[triali] < 0:
                        mem_angles[triali] += 180
                    if mem_angles[triali] > 180:
                        mem_angles[triali] -= 180
        
        #Add new randomized sequence of Gabor angles to dataframe.
        df_trialSeq.insert(loc = len(df_trialSeq.columns), column = 'MemItem_angle', value=mem_angles)
    
    """
    VI
    """
    #Assign colors to templates
    
    #Draw color combinations from csv.
    colors = genfromtxt(path_colors, delimiter=',')
    
    if training != True:
        #with one color combination for each miniblock, combinations 
        #just need to be shuffled and added to the dataframe.
        if len(colors) == 36:
            shuffle(colors)
            full_colors = np.repeat(colors, n_angle_bins, axis = 0)
        
        #with fewer color combinations than miniblocks:
        if len(colors) == 12:
            full_colors = np.zeros((len(df_trialSeq),2))
            
            m = 0
            #Iterate as often as the number of times that color combinations
            #have to be repeated within one miniblock (in the case of 12 combinations: 3).
            for i in range(int((len(df_trialSeq)/(len(colors)*n_angle_bins)))):
                for j in range(len(colors)):
                    #Randomize which of the two colors is shown first
                    rand_int = np.random.randint(1,3)
                    if rand_int == 1:
                        first = colors[j][0]
                        second = colors[j][1]
                        colors[j][0] = second
                        colors[j][1] = first
                
                #Shuffle smaller number of combinations and add them the same way as above.
                shuffle(colors)
                full_colors[m:m+(len(colors)*n_angle_bins),:] = np.repeat(colors,n_angle_bins,axis = 0)
                
                m += len(colors)*n_angle_bins
            
            #Preventing the same color combination from occurring twice in a row:
            for i in range(len(df_trialSeq)):
                if i%(len(colors)*n_angle_bins) == 0 and i != 0:
                    while full_colors[i][0] == full_colors[i-1][0] or full_colors[i][1] == full_colors[i-1][1]:
                        for j in range(len(colors)):
                            rand_int = np.random.randint(1,3)
                            if rand_int == 1:
                                first = colors[j][0]
                                second = colors[j][1]
                                colors[j][0] = second
                                colors[j][1] = first
                        shuffle(colors)
                        full_colors[i:i+(len(colors)*n_angle_bins),:] = np.repeat(colors, n_angle_bins, axis = 0)
    
    else:
        shuffle(colors)
        training_colors = colors[0:n_miniblocks_per_pass,:]
        full_colors = np.zeros((len(df_trialSeq),2))
        m = 0
        for i in range(n_miniblocks_per_pass):
            full_colors[m:m+n_angle_bins] = np.repeat(training_colors[i,:].reshape(1,2), n_angle_bins, axis = 0)
            m += n_angle_bins
    
    #Add two rows with unique sequene of set color combinations to dataframe.
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'Templ1_color', value = full_colors[:,0])
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'Templ2_color', value = full_colors[:,1])
        
    """
    VII
    """
    #Add row denoting whether "clockwise"" or "counterclockwise" will appear at top of screen
    #Same procedure as for active/latent assignment
    
    #Test condition testing whether assignment switching adds noise: assignment predictable for first half of experiment
    if test_button_assignment_shuffle:
        ccw_up = np.zeros((len(df_trialSeq)))
        n_condit_half = int(n_conditions/2)
        
        for i in range(len(ccw_up)):
            if i < n_condit_half:
                ccw_up[i] = 1
            else:
                cw_up = np.zeros(((int(np.round(n_condit_half/2))),1))
                ccw_uppp = np.ones(((int(np.round(n_condit_half/2))),1))
                
                cwccw_up = np.concatenate((cw_up,ccw_uppp))
                
                shuffle(cwccw_up)
                
                ccw_up[i] = cwccw_up[i-int(np.round(n_condit_half))]
                
        #Add column denoting cue presentation in judgment task to dataframe.
        df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'ccw_up', value = ccw_up)
    
    #Mostly-used condition with full randomization.
    else:
        n_condit_half = int(n_conditions/2)
        cw_up = np.zeros((n_condit_half,1))
        ccw_up = np.ones((n_condit_half,1))
        
        cwccw_up = np.concatenate((cw_up, ccw_up))
        shuffle(cwccw_up)
        
        #Add column denoting cue presentation in judgment task to dataframe.
        df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'ccw_up', value = cwccw_up)  # add shuffled 1s and 2s to dataframe
    
    """
    VIII
    """
    
    #Mark the trials where the two templates frame the memory item.
    
    frame_cond = np.zeros((n_conditions))
    
    for ind in df_trialSeq.index:
        #Extract distance of both templates from Gabor
        diff_templ1_memitem = direction_extent_of_angle_error_cwccw(df_trialSeq.loc[ind,'MemTempl1_angle'],df_trialSeq.loc[ind,'MemItem_angle'])             
        diff_templ2_memitem = direction_extent_of_angle_error_cwccw(df_trialSeq.loc[ind,'MemTempl2_angle'],df_trialSeq.loc[ind,'MemItem_angle'])
    
        #Determine whether the angles between each have different directions.
        if (diff_templ1_memitem < 0 and diff_templ2_memitem > 0) or (diff_templ1_memitem > 0 and diff_templ2_memitem < 0):
            frame_cond[ind] = 1
        if diff_templ1_memitem == 90 or diff_templ2_memitem == 90:
            frame_cond[ind] = 2
            
    #Add row denoting whether templates frame Gabor to dataframe.
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column='MemItem_framed', value=frame_cond)
    
    """
    (v)
    """
    
    #Determine angular distance between subsequent Gabor stimuli 
    #in the case of independently generated absolute orientations.
    
    if angleDef == 'cb_angles':
        dist = np.zeros((n_conditions))
        for i in range(len(df_trialSeq)):
            if i == 0 or i%n_angle_bins == 0:
                dist[i] = np.nan
            else:
                dist[i] = df_trialSeq['MemItem_angle'][i] - df_trialSeq['MemItem_angle'][i-1]
        
        df_trialSeq.insert(loc=len(df_trialSeq.columns),column='MemItem_deltangle',value=dist)
    
    """
    IX
    """
    
    #Randomize inter-trial intervals
    
    iti = np.zeros(len(df_trialSeq))
    for triali in range(len(df_trialSeq)):
        iti[triali] = np.random.choice(time_iti)
    
    df_trialSeq.insert(loc=len(df_trialSeq.columns),column='ITI',value=iti)
    
    """
    X
    """
    
    ##Add column marking trials in which templates are shown (first of miniblock).
    
    mb_start = np.zeros((len(df_trialSeq)))
    
    for m in range(len(mb_start)):
        if m == 0 or m%n_angle_bins == 0:
            mb_start[m] = 1
            
    df_trialSeq.insert(loc=len(df_trialSeq.columns),column='miniblock_start',value=mb_start)
    
    ##Add column marking trials where templates are reproduced a second time (last of miniblock).
    
    mb_end = np.zeros((len(df_trialSeq)))
    
    for m in range(len(mb_end)):
        if m < (len(mb_end)-1):
            if m != 0 and df_trialSeq.miniblock_start[m+1] == 1:
                mb_end[m] = 1
        else:
            mb_end[m] = 1
            
    df_trialSeq.insert(loc=len(df_trialSeq.columns),column='miniblock_end',value=mb_end)

    """
    XI
    """

    ##Select trials in which judgment task is performed pseudo-randomly (same amount in each miniblock).
    
    if probe == '33%':
        #Add new column to dataframe.
        cwccw_task = np.zeros((len(df_trialSeq)))
        df_trialSeq.insert(loc=len(df_trialSeq.columns), column='memory_probed',value= cwccw_task)
        
        #Select all trials in which the Gabor is framed by templates.
        for triali in range(n_conditions):
            if df_trialSeq['MemItem_framed'][triali] == 1:
                df_trialSeq['memory_probed'][triali] = 1
        
        #Iterate through miniblocks.
        for i in range(n_miniblocks_per_pass):
            sel = df_trialSeq.loc[df_trialSeq['Templ_index'] == i]
            
            #Determine how many trials were marked for judgment task in the first step.
            mp = np.count_nonzero(sel['memory_probed']==1)
            
            #Adjust this number until it is exactly 6 (=33% of 18).
            while mp != 6:
                #If there are less than 6 trials, randomly select a trial in the dataframe to turn to 1.
                if mp < 6: 
                    for triali in range((sel.index[0]),(sel.index[0]+(n_angle_bins+1))):
                        rand_index = np.random.randint((sel.index[0]),(sel.index[0]+(n_angle_bins+1)))
                        
                        if triali == rand_index:
                            df_trialSeq['memory_probed'][rand_index] = 1
                        
                            #Determine new count of selected trials.
                            mp = np.count_nonzero(df_trialSeq.loc[df_trialSeq['Templ_index'] == i]['memory_probed']==1)
                
                #If there are more than 6 trials, randomly select a trial in the dataframe to turn to 0.
                elif mp > 6: 
                    for triali in range((sel.index[0]),(sel.index[0]+(n_angle_bins+1))):
                        rand_index = np.random.randint((sel.index[0]),(sel.index[0]+(n_angle_bins+1)))
                        
                        if triali == rand_index:
                            df_trialSeq['memory_probed'][rand_index] = 0 
                            
                            #Determine new count of selected trials.
                            mp = np.count_nonzero(df_trialSeq.loc[df_trialSeq['Templ_index'] == i]['memory_probed']==1)
    
    ##Select all trials for judgment task.
    elif probe == 'always':
        cwccw_task = np.ones((len(df_trialSeq)))
        df_trialSeq.insert(loc=len(df_trialSeq.columns), column='memory_probed',value= cwccw_task)
    
    """
    XII
    """
    
    ##Fully randomize which template is cued for reproduction first.
    
    n_condit_half = int(n_template1_bins*n_template2_bins/2)
    
    t2_first = np.zeros((n_condit_half,1))
    t1_first = np.ones((n_condit_half,1))
    
    first_templ_1 = np.concatenate((t2_first, t1_first)).astype(int)
    first_templ_2 = np.concatenate((t2_first, t1_first)).astype(int)
    
    shuffle(first_templ_1)
    shuffle(first_templ_2)
    
    to_add_1 = np.zeros((len(df_trialSeq)))
    to_add_2 = np.zeros((len(df_trialSeq)))
    
    m = 0
    n = 0
    
    for i in range(len(df_trialSeq)):
        if i == 0 or i%n_angle_bins == 0:
            to_add_1[i] = first_templ_1[m]
            to_add_2[i] = np.nan
            m += 1
        elif (i != 0 and (i+1)%n_angle_bins == 0) or i == len(df_trialSeq)-1:
            to_add_2[i] = first_templ_2[n]
            to_add_1[i] = np.nan
            n += 1
        else:
            to_add_1[i] = np.nan
            to_add_2[i] = np.nan
    
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'Repr_1_Templ1_cue_first', value = to_add_1)  # add shuffled 1s and 2s to dataframe
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'Repr_2_Templ1_cue_first', value = to_add_2)  # add shuffled 1s and 2s to dataframe
    
    """
    XIII
    """
    ##Sequentially number miniblocks.
    
    num_mblock = np.repeat(np.arange(n_template1_bins*n_template2_bins),n_angle_bins)
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'mblock_no', value = num_mblock)
    
    """
    XIV
    """
    ##Add row denoting correct answers for each trial in judgment task.
    cor_ans = determine_correct_answers_cwccw(df_trialSeq)
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'correct_cwccw', value = cor_ans)
    
    """
    XV
    """
    ##Prepare rows to be filled during/after experiment.
        
    #add row of nan to insert subject's First Reproduction Angle of Template 1
    templ1_rep_1 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column='Templ1_Repr_1', value=templ1_rep_1)
    
    #add row of nan to insert subject's First RT of Template 1 reproduction
    templ1_RT_1 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ1_RT_1', value = templ1_RT_1)
    
    #add row of nan to insert subject's First Reproduction Angle of Template 2
    templ2_rep_1 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column = 'Templ2_Repr_1', value = templ2_rep_1)
    
    #add row of nan to insert subject's First RT of Template 2 reproduction
    templ2_RT_1 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ2_RT_1', value = templ2_RT_1)
    
    #add row of nan to insert subject's Second Reproduction Angle of Template 1
    templ1_rep_2 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ1_Repr_2', value = templ1_rep_2)
    
    #add row of nan to insert subject's Second RT of Template 1 reproduction
    templ1_RT_2 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ1_RT_2', value = templ1_RT_2)
    
    #add row of nan to insert subject's Second Reproduction Angle of Template 2
    templ2_rep_2 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ2_Repr_2', value = templ2_rep_2)
    
    #add row of nan to insert subject's Second RT of Template 2 reproduction
    templ2_RT_2 = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='Templ2_RT_2', value = templ2_RT_2)
    
    #add row of nan to insert subject's Reproduction Angle of Memory Item
    memitem_rep = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='MemItem_Repr', value = memitem_rep)
    
    #add row of nan to insert subject's RT of Memory Item reproduction
    memitem_RT = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='MemItem_RT', value = memitem_RT)
    
    #add row of zeros to insert subject's response to cwccw task
    cwccw_task = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns), column ='cwccw_resp', value = cwccw_task)
    
    #add row of zeros to insert subject's reaction time to cwccw task
    cwccw_rt = np.zeros((len(df_trialSeq)))
    df_trialSeq.insert(loc=len(df_trialSeq.columns),column ='cwccw_rt', value = cwccw_rt)
    
    #add column to add original response bar orientation
    df_trialSeq['orig_respbar_memitem'] = np.zeros((len(df_trialSeq)))
    
    #add columns to add original response bar orientation
    df_trialSeq['orig_respbar_templ1_Repr_1'] = np.zeros((len(df_trialSeq)))
    df_trialSeq['orig_respbar_templ1_Repr_2'] = np.zeros((len(df_trialSeq)))
    
    #add columns to add original response bar orientation
    df_trialSeq['orig_respbar_templ2_Repr_1'] = np.zeros((len(df_trialSeq)))
    df_trialSeq['orig_respbar_templ2_Repr_2'] = np.zeros((len(df_trialSeq)))
        
    return df_trialSeq

#%%
def training_prepTrials():
    """
    Function to generate trial sequence for angle training.
    Builds long dataframe to play through all possible relative orientations
    at least once, while having a great spare number of trials for different lengths needed.
    Generating 360 trials with an initial need for 30.
    
    """
    blocks = 20
    
    angle_bins = np.linspace(-90,90,19)
    orient_bins = (np.linspace(0,180,len(angle_bins)))
    
    template_1 = np.zeros(((len(angle_bins)-1)*blocks))
    template_2 = np.zeros(((len(angle_bins)-1)*blocks))
    angles = np.zeros(((len(angle_bins)-1)*blocks))

    #One block contains all possible relative orientations
    for blocki in range(blocks):
        z= 0
        angles_small = np.zeros((len(angle_bins)-1))
        
        #Iterate through all relative orientation bins instead of 90°
        for bini in range(len(angle_bins)):
            if angle_bins[bini] != 90:
                #Add random number in relative orientation bin to array
                  while angles_small[z] == 0 or angles_small[z] == 90 or angles_small[z] == -90:
                      angles_small[z] = np.random.randint(angle_bins[bini],angle_bins[bini+1])
            z+=1

        #Shuffle array to randomize sequence of relative orientation bins
        shuffle(angles_small)
        
        #Add randomly chosen and ordered sequence of current block to dataframe.
        angles[int(blocki*(len(angle_bins)-1)):int(blocki*(len(angle_bins)-1)+len(angle_bins)-1)] = angles_small
    
    #Do the same for the absolute orientation of the first stimulus.
    for blocki in range(blocks):
        z=0
        template1_small = np.zeros((len(orient_bins)-1))
        for bini in range(len(orient_bins)):
            if orient_bins[bini] != 180:
                while template1_small[z] == 0 or template1_small[z] == 90:
                    template1_small[z] = np.random.randint(orient_bins[bini],orient_bins[bini+1])

            z+=1
    
        shuffle(template1_small)
        template_1[int(blocki*(len(orient_bins)-1)):int(blocki*(len(orient_bins)-1)+len(orient_bins)-1)] = template1_small
    
    #The absolute orientation of the second stimulus is determined from the 
    #absolute orientation of the first and the relative orientation between the two
    for i in range(len(template_2)):
        template_2[i] = template_1[i] + angles[i]
        if template_2[i] > 180:
            template_2[i] = template_2[i] - 180
        if template_2[i] < 0:
            template_2[i] = template_2[i] + 180
    
    #Add all three arrays to dataframe.
    template_1 = template_1.reshape(len(template_1),1)
    template_2 = template_2.reshape(len(template_2),1)
    angles = angles.reshape(len(angles),1)
    
    all_tog = np.concatenate((template_1, angles, template_2),axis=1)
    df_training = pd.DataFrame(all_tog,columns=['template_1','angles','template_2'])
    
    #Add column of random 1s and 0s to denote if ccw is presented on top of screen in any given trial
    n_condit_half = int(len(df_training)/2)
    cw_up = np.zeros((n_condit_half,1))
    ccw_up = np.ones((n_condit_half,1))
    
    cwccw_up = np.concatenate((cw_up, ccw_up))
    shuffle(cwccw_up)
    
    df_training.insert(loc=len(df_training.columns), column = 'ccw_up', value = cwccw_up)

    #Add column denoting correct answers
    cor_ans = determine_correct_answers_cwccw(df_training, angletraining = 1)
    df_training.insert(loc=len(df_training.columns), column = 'correct_answer', value = cor_ans)
    
    #Add empty column to add participant response
    cwccw_resp = np.zeros((len(df_training)))
    df_training.insert(loc=len(df_training.columns), column ='cwccw_resp', value = cwccw_resp)
    
    #Add empty column to add participant response time
    cwccw_time = np.zeros((len(df_training)))
    df_training.insert(loc=len(df_training.columns), column ='cwccw_RT', value = cwccw_time)
    
    return df_training

df = training_prepTrials()