#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: Main script to run the behavioral paradigm
#Author: Ilona Vieten (adapted from Darinka Truebutschek)
#Date created: 10/05/2021
#Date last modified: 08/04/2022
#Python version: 3.6.13

### Import needed tools ###
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Set preferences for psychopy
from psychopy import prefs
prefs.general['audioLib'] = 'ptb' #needed on mac to allow shutdown of experiment
prefs.general['units'] = 'deg'

from psychopy import core, data, event, gui, logging, monitors, visual

### Import experimental parameters ###
from MemTempl_config import *
from MemTempl_trialsequence import SerDep_prepTrials
from MemTempl_Taskfunctions import *
from MemTempl_eyetracker import setupEyetracker, calibrateEyetracker, quitEyetracker

### Display settings ###
print(prefs)

'''
====================================================
Store info about the experimental session
====================================================
'''

#Show a dialog box to enter session information
exp_name = 'SerDep_MemTemp'
exp_info = {'subID': '',
            'sessID': ''}

dlg = gui.DlgFromDict(dictionary=exp_info, order=['subID', 'sessID'], title=exp_name) # by doing this, the exp_info dictionary is completed with subID and sessID inputs

#If 'cancel' is pressed, quit
if dlg.OK == False:
    core.quit()

#store additional info about experiment
exp_info['exp_name'] = exp_name
exp_info['date'] = data.getDateStr()

'''
====================================================
Specify exit options to quit full screen if need be (especially during programming)
====================================================
'''

event.globalKeys.clear()
event.globalKeys.add(key='a', func=os._exit, func_args=[1], func_kwargs=None) #add a key that forcibly exits the script (in this case, a)

'''
====================================================
Prepare trials
====================================================
'''
##Create trial sequence according to whether 
#the same is going to be performed once or twice (both sessions in one or two separate experiments)

#initial generation of trial sequence
if prepTrials:
    tmp_1 = SerDep_prepTrials()
    
    #If both sessions are performed in one experiment, sequence is copied and concatenated together
    if n_passes == 2:
        trialSeq = []
        tmp_2 = tmp_1.copy()
        
        pass_no_1 = np.ones((len(tmp_1),1))
        tmp_1.insert(0, 'Pass_no', pass_no_1)
        trialSeq.append(tmp_1)
        
        pass_no_2 = np.full(((len(tmp_2)), 1), 2)
        tmp_2.insert(0, 'Pass_no', pass_no_2)
        trialSeq.append(tmp_2)
    
        trialSeq = pd.concat(trialSeq)
        trialSeq = trialSeq.reset_index(drop=True)
    
    #If only one session is performed in one experiment,
    #trial sequence is either used as is (in first experiment)
    #or generated from existing csv of first experiment (in second experiment)
    if n_passes == 1:
        #Input which experiment is being performed
        if exp_info['sessID'] == '001' or exp_info['sessID'] == '000':
            trialSeq = tmp_1
            print('trials generated 000/001')
            
        elif exp_info['sessID'] == '002':
            file = 'rawDat_Subj_' + str(exp_info['subID']) + '_Sess_001_final.csv'
            file_name = path_rawDat + file
            trialSeq = pd.read_csv(file_name, delimiter = ',')
            print('trials generated 002')

if prepTrials:
    #Plot distribution for angle bins
    bins = np.linspace(0, n_angle_bins, n_angle_bins+1)
    plt.hist(trialSeq.MemItem_bin, bins=bins)
    plt.title('Angle bins')
    plt.show()
        
    plt.hist(trialSeq.MemItem_bin[trialSeq.Active == 1], bins=bins)
    plt.title('Angle bins for first template active')
    plt.show()

    plt.hist(trialSeq.MemItem_bin[trialSeq.Active == 2], bins=bins)
    plt.title('Angle bins for second template active')
    plt.show()

    #Plot distribution for actual memory item orientations
    bins = np.linspace(0,180,n_angle_bins+1)
    bins = bins+5
    plt.hist(trialSeq.MemItem_angle, bins=bins)
    plt.title('Memory Item orientations')
    plt.show()

    plt.hist(trialSeq.MemItem_angle[trialSeq.Active == 1], bins=bins)
    plt.title('Memory Item orientations for first template active')
    plt.show()

    plt.hist(trialSeq.MemItem_angle[trialSeq.Active == 2], bins=bins)
    plt.title('Memory Item orientations for second template active')
    plt.show()
    
    #Plot distribution for template orientations
    plt.hist(trialSeq.MemTempl1_angle, bins=np.linspace(5, 185,7))
    plt.title('Memory Template 1 orientations')
    plt.show()
    
    plt.hist(trialSeq.MemTempl2_angle, bins=np.linspace(5, 185,7))
    plt.title('Memory Template 2 orientations')
    plt.show()

    #Plot distributions for orientation differences between sequential memory items
    bins = np.linspace(-90, 90, n_angle_bins+1)
    plt.hist(trialSeq.MemItem_deltangle, bins=bins)
    plt.title('MemItem Orientation distances')
    plt.show()

    plt.hist(trialSeq.MemItem_deltangle[trialSeq.Active == 1], bins=bins)
    plt.title('MemItem Orientation distances for first template active')
    plt.show()

    plt.hist(trialSeq.MemItem_deltangle[trialSeq.Active == 2], bins=bins)
    plt.title('MemItem Orientation distances for second template active')
    plt.show()
    
    #Compute autocorrelation
    corr_angleBins = trialSeq.MemItem_bin.autocorr(lag=1)
    print('The  autocorrelation for angle bins at lag 1 is: ' + str(corr_angleBins))
    pd.plotting.autocorrelation_plot(trialSeq.MemItem_bin, ax=None)
    plt.show()

    corr_angle = trialSeq.MemItem_angle.autocorr(lag=1)
    print('The  autocorrelation for angles at lag 1 is: ' + str(corr_angle))
    pd.plotting.autocorrelation_plot(trialSeq.MemItem_angle, ax=None)
    plt.show()

    corr_deltaAngle = trialSeq.MemItem_deltangle.autocorr(lag=1)
    print('The  autocorrelation for delta angle at lag 1 is: ' + str(corr_deltaAngle))
    pd.plotting.autocorrelation_plot(trialSeq.MemItem_deltangle, ax=None)
    plt.show()

        
'''
====================================================
Create logs
====================================================
'''

#Prepare filenames and directories for info-, log-, and raw Data for subject
info_filename = '/info_Subj_' + str(exp_info['subID']) + '_Sess_' +  str(exp_info['sessID'])
info_filename = path_rawDat + info_filename #path_rawDat denotes the general target folder, and then info_filename is put into that
log_filename = '/log_Subj_' + str(exp_info['subID']) + '_Sess_' + str(exp_info['sessID']) # log file records things that happen on the system/computer during experiment
log_filename = path_rawDat + log_filename
rawDat_filename = '/rawDat_Subj_' + str(exp_info['subID']) + '_Sess_' +  str(exp_info['sessID'])
rawDat_filename = path_rawDat + rawDat_filename

#Save a logfile for detailed info
log_all = logging.LogFile(str(log_filename) + '_logAll.log', level=logging.DEBUG)
log_warning = logging.LogFile()

# Save global experiment info
file_pickle = str(info_filename) + '.pickle' #info_filename is saved as .pickle
pickle.dump(exp_info, open(file_pickle, 'wb'))

'''
====================================================
Create components
====================================================
'''

#Set monitor properties
my_monitor = monitors.Monitor('ExpPres', width=mon_width, distance=mon_distance)
my_monitor.setSizePix(mon_size)
my_monitor.saveMon()

#Set window properties
if fullscr == True:
    win = visual.Window(fullscr=True, monitor=my_monitor,  size=mon_size, units='deg', 
                        colorSpace='rgb255', color=screen_color, allowGUI=False, useRetina=True) # window opens here
else:
    win = visual.Window(fullscr=False, monitor=my_monitor,  size=mon_size*0.75, units='deg',  # this is so a non-fullscreen window doesn't still fill the whole monitor due to its size
                        colorSpace='rgb255', color=screen_color, allowGUI=False, useRetina=True) # window opens here

measuredFrame = win.getActualFrameRate

'''
====================================================
Set up eyetracker
====================================================
'''
if eyetracker:
    tk, filename_edf = setupEyetracker(exp_info['subID'], exp_info['sessID'], win, mon_size)

'''
====================================================
Make sure data is saved in case experiment is aborted
====================================================
'''

def globalQuit():
    #Save data
    rawDat_filename_aborted = str(rawDat_filename) + '_aborted.pkl' #save as pickle
    trialSeq.to_pickle(rawDat_filename_aborted) #save current state of experiment as pickle

    rawDat_filename_aborted_csv = str(rawDat_filename) + '_aborted.csv' #save as csv
    trialSeq.to_csv(rawDat_filename_aborted_csv) #same thing as csv
    
    #Save eyetracking data
    if eyetracker:
        filename_edf_aborted = filename_edf + '_aborted'
        quitEyetracker(tk=tk, edf_filename=filename_edf_aborted, win=win, path_edfDat=path_edfDat)

    #Quit
    win.flip()
    win.close()
    core.quit()

event.globalKeys.add(key='q', func=globalQuit)


'''
====================================================
Run experiment
====================================================
'''
#Welcome & instructions screen
run_text(win, inst_beginning, text_size, text_font, text_color, duration='inf', button='return')

#Loop over passes (only if both are performed in one experiment) and trials
for passi in np.arange(n_passes):
    #Calibrate eyetracker & start recording
    if eyetracker:
        tk = calibrateEyetracker(tk=tk, win=win)
        error = tk.startRecording(1, 1, 1, 1)
        
    #Calculate trial number
    if passi == 0:
        n_trials_start = 0
        n_trials_stop = n_trials_per_pass
    else:
        n_trials_start = 0+(n_trials_per_pass*1)
        n_trials_stop = n_trials_start+n_trials_per_pass
    
    #Run countdown
    if passi == 0:
        run_text(win, ready_5, text_size, text_font, text_color, duration=time_countdown, button=None)
        run_text(win, ready_4, text_size, text_font, text_color, duration=time_countdown, button=None)
        run_text(win, ready_3, text_size, text_font, text_color, duration=time_countdown, button=None)
        run_text(win, ready_2, text_size, text_font, text_color, duration=time_countdown, button=None)
        run_text(win, ready_1, text_size, text_font, text_color, duration=time_countdown, button=None)
    
    #Send start of pass to eyetracker
    if eyetracker:
        tk.sendMessage('Start of pass {}'.format(passi))
    
    #Run fixation cross
    run_fix(win, fix_size, fix_lineWidth, fix_lineColor, duration=0.5, fix_value=fix_value, colorspace=templ_colorspace)
    
    for triali in np.linspace(n_trials_start, n_trials_stop-1, n_trials_per_pass):
        triali = int(triali)
        if triali == 0:
            last_break_triali = 0
        
        #Recalibrate eyetracker if previous trial had a long pause
        if eyetracker and recalibrate:
            tk = calibrateEyetracker(tk=tk, win=win)
            
        #Start recording eyetracking data
        if eyetracker:
            error = tk.startRecording(1, 1, 1, 1)
        
        print('Trial ' + str(triali)) #not needed if only one monitor
        
        ##Retrieve all necessary information for the trial from trial sequence dataframe
        #Initialize variables
        templ1_first_repr1 = False
        templ1_first_repr2 = False
        
        #Retrieve trial type (start, middle, end of miniblock), 
        #template orientations, and which template is reproduced first
        if trialSeq.miniblock_start[triali] == 1: 
            
            #Variable denoting that it is the first trial of the miniblock
            rem_templ = True
            trig_miniblock = triggers_miniblock[0]
            
            #Variable containing template 1 orientation
            templ1_orient = trialSeq.MemTempl1_angle[triali]
            trig_templ1 = triggers_template1[int(trialSeq.MemTempl1_bin[triali])]
            
            #Variable containing template 2 orientation
            templ2_orient = trialSeq.MemTempl2_angle[triali]
            trig_templ2 = triggers_template2[int(trialSeq.MemTempl2_bin[triali])]
            
            #Variable denoting whether first template is cued to be reproduced first
            if trialSeq.Repr_1_Templ1_cue_first[triali] == 1:
                templ1_first_repr1 = True
            else:
                templ1_first_repr1 = False
        else:
            rem_templ = False
        
        #Same for last trial of miniblock
        if trialSeq.miniblock_end[triali] == 1:
            
            #Variable denoting that it is the last trial of the miniblock
            end_rememb = True
            trig_miniblock = triggers_miniblock[1]
            
            #Variable denoting whether first template is cued to be reproduced first
            if trialSeq.Repr_2_Templ1_cue_first[triali] == 1:
                templ1_first_repr2 = True
            else:
                templ1_first_repr2 = False
        else:
            end_rememb = False
        
        #Determining which template is to be used for judgment task (i.e., which is "active")
        if trialSeq.Active[triali] == 1:
            trig_cue = triggers_cue[0]
        else:
            trig_cue = triggers_cue[1]
        
        #Determining whether trial contains judgment task
        if trialSeq.memory_probed[triali] == 1:
            cwccw_task = True
            trig_probe = triggers_probe[1]
        else:
            cwccw_task = False
            trig_probe = triggers_probe[0]
        
        #Determining whether trial is followed by short break
        if trialSeq.Short_Break[triali] == 1:
            break_trial = True
            trig_break = triggers_break[1]
        else:
            break_trial = False
            trig_break = triggers_break[0]
        
        #Determining whether trial is followed by longer pause
        if trialSeq.Long_Break[triali] == 1:
            pause_trial = True
            trig_pause = triggers_pause[1]
            
            #In the case of a long break, eye tracker is recalibrated afterwards
            recalibrate = True
        else:
            pause_trial = False
            trig_pause = triggers_pause[0]
            recalibrate = False
        
        #Determining whether trial is neither at the start nor the end of a miniblock
        if trialSeq.miniblock_start[triali] == 0 and trialSeq.miniblock_end[triali] == 0:
            trig_miniblock = triggers_miniblock[2]
        
        #Initialize array of possible initial orientations of response bar
        pool = np.arange(180)
        
        #Determining which template is used for judgment task (is "active")
        trial_active = trialSeq.Active[triali]
        
        #Determining button assignment (counterclockwise cue to be presented at top of screen?)
        trial_ccw_up = trialSeq.ccw_up[triali]
        
        #Determining Gabor orientation
        trial_gabor = trialSeq.MemItem_angle[triali]
        trig_gabor = triggers_gabor[int(trialSeq.MemItem_bin[triali])]
        
        #Determining template colors
        templ1_color = trialSeq.Templ1_color[triali]
        templ2_color = trialSeq.Templ2_color[triali]
        
        #Determining inter-trial interval
        iti = trialSeq.ITI[triali]
        
        #Send start of trial to eyetracker
        if eyetracker:
            tk.sendMessage('Start of trial {}'.format(triali))
        
        #Run one trial
        if eyetracker:
            tracker=tk
        else:
            tracker=None
        
        last_break_triali,templ11_resp_name, templ11_resp_rt, templ11_resp_duration, templ11_angle,templ21_resp_name, templ21_resp_rt, templ21_resp_duration, templ21_angle,mem_item_resp_name, mem_item_resp_rt, mem_item_resp_duration, mem_item_angle,templ12_resp_name, templ12_resp_rt, templ12_resp_duration, templ12_angle,templ22_resp_name, templ22_resp_rt, templ22_resp_duration, templ22_angle, cwccw_resp, cwccw_rt, resp_orient_templ1_1, resp_orient_templ2_1, resp_orient_templ1_2, resp_orient_templ2_2, resp_orient_item = run_trial(
            win, bg_color=screen_color, break_trial = break_trial, pause_trial = pause_trial, break_text_short = exp_break_short, break_text_long = exp_break_long, text_size = text_size, text_color = text_color, text_font = text_font, memory_probe_text_1 = memory_probe_text_1, memory_probe_text_2 = memory_probe_text_2, trial_active=trial_active, trial_ccw_up = trial_ccw_up, inst_reproduction= inst_reproduction, templ_instr_text_1 = inst_templates_1, templ_instr_text_2 = inst_templates_2, post_templ_text = inst_templates_3, cue_size=cue_size, 
            cue_units=cue_units, templ1_cue_first_repr1 = templ1_first_repr1, templ1_cue_first_repr2 = templ1_first_repr2,
            gabor_tex=gabor_tex, gabor_mask=gabor_mask, gabor_units=gabor_units, gabor_sf=gabor_sf, gabor_phase=gabor_phase, gabor_contrast=gabor_contrast, gabor_maskParams=gabor_maskSD, gabor_orient=trial_gabor, gabor_size=gabor_size, gabor_duration=time_gabor,
            mask_noiseType=mask_noiseType, mask_mask=mask_mask, mask_units=mask_units, mask_noiseElementSize=mask_noiseElementSize, mask_size=mask_size, mask_duration=time_mask, mask_contrast = mask_contrast,
            fix_size=fix_size, fix_lineWidth=fix_lineWidth, fix_lineColor=fix_lineColor, fix_duration=time_del, fix_value = fix_value,
            resp_width=respBar_width, resp_height=respBar_height, resp_units=respBar_units, respBarpool = pool, respBar_color = respBar_color, resp_updateKeys=[keysCW_large, keysCCW_large], resp_logKeys=keysLogResp, keyscwccw = [keysup,keysdown],
            iti_duration=iti,
            eyetracking=eyetracker, tracker=tracker,
            triggersCue=trig_cue, triggersGabor=trig_gabor, triggersMiniblock = trig_miniblock, triggersProbe = trig_probe, triggersTempl1 = trig_templ1, triggersTempl2 = trig_templ2, triggersBreak = trig_break, triggersPause = trig_pause,
            rem_templ=rem_templ, end_rememb=end_rememb, templ_width=templ_width, templ_height=templ_height,templ_units=templ_units,templ_duration= time_templ,templ1_orient=templ1_orient, templ2_orient=templ2_orient, templ_colorspace = templ_colorspace, templ1_color = templ1_color, templ2_color = templ2_color, templ_value = templ_value,
            cwccw_task = cwccw_task, trialSeq = trialSeq, triali = triali, feedback_1 = angletraining_feedback_1, feedback_2 = angletraining_feedback_2, last_break_triali = last_break_triali,rawDat_filename = rawDat_filename,training=training)
        
        #Send end of trial to eyetracker
        if eyetracker:
            tk.sendMessage('End of trial')

        ##Save data in case experiment has been aborted
        #Template 1 First reproduction
        trialSeq['Templ1_RT_1'] = trialSeq['Templ1_RT_1'].astype(object)
        trialSeq.at[triali,'Templ1_Repr_1'] = templ11_angle
        trialSeq.at[triali, 'Templ1_RT_1'] = templ11_resp_rt

        #Template 2 First reproduction
        trialSeq['Templ2_RT_1'] = trialSeq['Templ2_RT_1'].astype(object)
        trialSeq.at[triali,'Templ2_Repr_1'] = templ21_angle
        trialSeq.at[triali, 'Templ2_RT_1'] = templ21_resp_rt

        #Memory Item reproduction
        trialSeq['MemItem_RT'] = trialSeq['MemItem_RT'].astype(object)
        trialSeq.at[triali,'MemItem_Repr'] = mem_item_angle
        trialSeq.at[triali, 'MemItem_RT'] = mem_item_resp_rt

        #Template 1 Second reproduction
        trialSeq['Templ1_RT_2'] = trialSeq['Templ1_RT_2'].astype(object)
        trialSeq.at[triali,'Templ1_Repr_2'] = templ12_angle
        trialSeq.at[triali, 'Templ1_RT_2'] = templ12_resp_rt

        #Template 2 Second reproduction
        trialSeq['Templ2_RT_2'] = trialSeq['Templ2_RT_2'].astype(object)
        trialSeq.at[triali,'Templ2_Repr_2'] = templ22_angle
        trialSeq.at[triali, 'Templ2_RT_2'] = templ22_resp_rt
        
        #cwccw-task response and reaction time
        trialSeq.at[triali,'cwccw_resp'] = cwccw_resp
        trialSeq.at[triali,'cwccw_rt'] = cwccw_rt
        
        #initial response bar orientations
        trialSeq.at[triali, 'orig_respbar_memitem'] = resp_orient_item
        trialSeq.at[triali, 'orig_respbar_templ1_Repr_1'] = resp_orient_templ1_1
        trialSeq.at[triali, 'orig_respbar_templ2_Repr_1'] = resp_orient_templ2_1
        trialSeq.at[triali, 'orig_respbar_templ1_Repr_2'] = resp_orient_templ1_2
        trialSeq.at[triali, 'orig_respbar_templ2_Repr_2'] = resp_orient_templ2_2
        
    #Save end of block to eyetracker & stop recording
    if eyetracker:
        tk.sendMessage('End of block')
        tk.stopRecording()

    print("pass " + str(passi) + " completed")
        
#Save final data at the end of the experiment
if training:
    rawDat_filename_pickle = str(rawDat_filename) + '_training.pkl' #save as pickle
    trialSeq.to_pickle(rawDat_filename_pickle)

    rawDat_filename_csv = str(rawDat_filename) + '_training.csv' #save as csv
    trialSeq.to_csv(rawDat_filename_csv)
else:
    rawDat_filename_pickle = str(rawDat_filename) + '_final.pkl' #save as pickle
    trialSeq.to_pickle(rawDat_filename_pickle)

    rawDat_filename_csv = str(rawDat_filename) + '_final.csv' #save as csv
    trialSeq.to_csv(rawDat_filename_csv)

if eyetracker:
    filename_edf_final = filename_edf #str(filename_edf) + '_final.edf'
    quitEyetracker(tk=tk, edf_filename=filename_edf_final, win=win, path_edfDat=path_edfDat)

#Final feedback (generated as in run_trial()) and Goodbye
if training == 0:
    ratio = experiment_ratio_correct(trialSeq[last_break_triali:])
        
    feedback_text = angletraining_feedback_1 + str(ratio) + angletraining_feedback_2
    
    run_text(win, feedback_text, text_size, text_font, text_color, duration='inf', button='return')
    run_text(win, goodbye, text_size, text_font, text_color, duration='inf', button='return')

else:
    ratio = experiment_ratio_correct(trialSeq[:])
        
    feedback_text = angletraining_feedback_1 + str(ratio) + angletraining_feedback_2
    
    run_text(win, feedback_text, text_size, text_font, text_color, duration='inf', button='return')
    run_text(win, goodbye_training, text_size, text_font, text_color, duration='inf', button='return')

print('Finished')

#Close session
win.flip()
win.close()
core.quit()