# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:49:56 2021

@author: ilona.vieten
"""

#Purpose: Main script to run the behavioral paradigm
#Author: Ilona Vieten
#Date created: 18/11/2021
#Date last modified: 18/11/2021
#Python version: 3.6.13

### Import needed tools ###
import os
import pickle

### Set preferences for psychopy
from psychopy import prefs
prefs.general['audioLib'] = 'ptb' #needed on mac to allow shutdown of experiment
prefs.general['units'] = 'deg'

from psychopy import core, data, event, gui, logging, monitors, visual

#from psychopy.hardware import keyboard
#kb = keyboard.Keyboard()

### Import experimental parameters ###
from MemTempl_config import *
from MemTempl_Taskfunctions import run_text, run_angletraining, angletraining_ratio_correct

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
Create logs
====================================================
'''

#Prepare filenames and directories for info-, log-, and raw Data for subject
info_filename = 'info_Subj_' + str(exp_info['subID']) + '_Sess_' +  str(exp_info['sessID']) + '_angletraining'
info_filename = path_rawDat + info_filename #path_rawDat denotes the general target folder, and then info_filename is put into that
log_filename = 'log_Subj_' + str(exp_info['subID']) + '_Sess_' + str(exp_info['sessID']) + '_angletraining' # log file records things that happen on the system/computer during experiment
log_filename = path_rawDat + log_filename
rawDat_filename = 'rawDat_Subj_' + str(exp_info['subID']) + '_Sess_' +  str(exp_info['sessID'] + '_angletraining')
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
Make sure data is saved in case experiment is aborted
====================================================
'''

def globalQuit():
    #Save data
    rawDat_filename_aborted = str(rawDat_filename) + '_aborted.pkl' #save as pickle
    df_training.to_pickle(rawDat_filename_aborted) #save current state of experiment as pickle

    rawDat_filename_aborted_csv = str(rawDat_filename) + '_aborted.csv' #save as csv
    df_training.to_csv(rawDat_filename_aborted_csv) #same thing as csv
    
    #Quit
    win.flip()
    win.close()
    core.quit()

event.globalKeys.add(key='q', func=globalQuit)

'''
====================================================
Run angletraining
====================================================
'''

df_training = run_angletraining(win, start_at_trial,length_initial_block,length_additional_block)


'''
====================================================
Do analysis and give feedback
====================================================

'''

#calculate performance after initial number of trials
ratio = angletraining_ratio_correct(df_training,((n*length_additional_block)+length_initial_block),rating_range,window=averaging_window)

#continue by displaying additional trials until ratio meets threshold
while ratio <= target_ratio:
    if n == 0:
        start_at_trial = length_initial_block
    else:
        start_at_trial = length_initial_block+(n*length_additional_block)
    
    df_training = run_angletraining(win, start_at_trial,length_initial_block,length_additional_block,df=df_training)

    print('n:')
    print(n)
    print('current last trial: ')
    print((n*length_additional_block)+length_initial_block)
    
    ratio = angletraining_ratio_correct(df_training,((n*length_additional_block)+length_initial_block),rating_range,window=averaging_window)

    print(ratio)
    n+=1
    
'''
====================================================
Conclude experimental run
====================================================

'''
#provide subject with feedback when threshold is reached
feedback_text = angletraining_feedback_1 + str(ratio) + angletraining_feedback_2
run_text(win, feedback_text, text_size, text_font, text_color, duration='inf', button='return')

#save dataframe as csv/pickle
rawDat_filename_pickle = str(rawDat_filename) + '.pkl' #save as pickle
df_training.to_pickle(rawDat_filename_pickle)

rawDat_filename_csv = str(rawDat_filename) + '.csv' #save as csv
df_training.to_csv(rawDat_filename_csv)

#Close session
win.flip()
win.close()
core.quit()