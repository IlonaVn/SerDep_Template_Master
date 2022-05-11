# -*- coding: utf-8 -*-

#Purpose: Config file to set/store all important parameters
#Author: Ilona Vieten (adapted from Darinka Truebutschek)
#Date created: 06/10/2021
#Date last modified: 21/04/2022
#Python version: 3.6.13

import numpy as np
from pathlib import Path
recalibrate = False

'''
====================================================
Define important variables
====================================================
'''

computer = 'laptop' #laptop, setup, lab, office, MPI; define computer on which to run experiment
eyetracker = 0 #0=no, 1=yes
language = 'English' #English or German

'''
====================================================
Establish initial experiment variables for angletraining
====================================================
'''

rating_range = np.concatenate((np.linspace(-75,-15,61),np.linspace(15,75,61)))
length_initial_block = 30
target_ratio = 85
length_additional_block = 1
averaging_window = 15

start_at_trial = 0
n = 0

'''
====================================================
Establish initial experiment variables for main task
====================================================
'''

prepTrials = 1 #if 0 = use dummy trial definition, if 1 = actually generate the trials
angleDef = 'cb_deltaAngles' #'cb_angles'#'cb_deltaAngles' #'cb_angles'#how are orientations drawn on each trial; 'cb_anlges': counterbalanced orientations; 'cb_deltaAngles': counterbalanced orientation differences 
training = 0 #is this a training session or not; if it is, show feedback screen
probe = 'always' #or '33%': whether cwccw task is triggered in every trial or in 33% of trials
both_passes = False
test_button_assignment_shuffle = False

if training:
    n_angle_bins = 18 #Number of angular bins from which to draw the memory stimulus (1 = 6-15; 2 = 16-25; etc.)
    n_template1_bins = 1
    n_template2_bins = 2
    n_miniblocks = n_template1_bins*n_template2_bins
    n_passes = 1
    n_miniblocks_per_pass = 2
    n_trials_per_miniblock = n_angle_bins
    n_trials_per_pass = n_miniblocks_per_pass * n_angle_bins
    n_miniblocks_before_short_break = n_miniblocks_per_pass
    n_miniblocks_before_long_break = 9
else:   
    n_angle_bins = 18 #Number of angular bins from which to draw the memory stimulus (1 = 6-15; 2 = 16-25; etc.)
    n_template1_bins = 6
    n_template2_bins = 6
    n_miniblocks = n_template1_bins*n_template2_bins
    n_miniblocks_per_pass = n_template1_bins*n_template2_bins
    n_trials_per_miniblock = n_angle_bins
    n_trials_per_pass = n_miniblocks_per_pass * n_trials_per_miniblock
    n_miniblocks_before_short_break = 3
    n_miniblocks_before_long_break = 9
    if both_passes == True: 
        n_passes = 2
    else:
        n_passes = 1
        
if computer == 'setup':
    fullscr=True
else:
    fullscr=False

if (computer != 'setup') & (computer == 'MPI'):
    keysCW_large = 'r' #right (arrow key)
    keysCW_small = 'e'
    keysCCW_large = 'l' #left (arrow key)
    keysCCW_small = 'k'
    keysLogResp = 'return' #key to confirm response
    keyscw = 'o'
    keyscountcw = 'w'
elif computer == 'setup' or computer == 'office' or computer == 'laptop':
    keysCW_large = "3"
    keysCCW_large = "1"
    keysLogResp = 'return'
    
    #doesn't change throughout script!
    keysup = '2'
    keysdown = '4'


### Path definitions ###
if computer == 'laptop':
    path_str = 'C:/Users/illil/Documents/01 Uni Göttingen/01 Masterarbeit/Master_Data/Experiment Memory Templates/Templates_Exp_Code_Cleaned'
    path_main = Path(path_str)
elif computer == 'office':
    path_main = Path('W:/ilona.vieten/Experiment Memory Templates/Templates_Exp_Code')
elif computer == 'setup':
    path_main = Path('C:/Users/Caspar/Desktop/ilona/Templates_Exp_Code')
elif computer == 'MPI':
    path_main = Path('P:/2021-0306-MemoryDistortions/MemTemplColor')


path_expScripts = path_main / 'ExpCode/' #experimental code
path_anScripts = path_main / 'AnCode/' #analysis code
path_rawDat = path_str + '/RawData/' #raw data
path_edfDat = path_main / 'EyetrackingData/' #raw eyetracking data
path_results = path_main / 'Results' 

#path to colors csv
if computer == 'office':
    path_colors = 'W:/ilona.vieten/Experiment Memory Templates/Templates_Exp_Code/Stimuli/colors_12.csv'
elif computer == 'laptop':
    path_colors = path_main / 'Stimuli/colors_12.csv'
elif computer == 'setup':
    path_colors = path_main / 'Stimuli/colors_12.csv'
elif computer  == 'MPI':
    path_colors = path_main / 'Stimuli/colors_12.csv'

### Experimental parameters ###

#Screen parameters
if computer == 'laptop':
    refresh_rate = 60 #in Hz (this is only an assumption for the mac)
    mon_size = np.array([1920, 1080]) #in pixels
    mon_width = 34.93 #in cm --> measure
    mon_height = 24.07 #in cm --> measure
    mon_distance = 57 #measure
    
elif computer == 'office':
    refresh_rate = 59 # checked, correct
    mon_size = np.array([1920, 1080]) #in pixels
    mon_width = 34.93 #in cm --> measure
    mon_height = 24.07 #in cm --> measure
    mon_distance = 57 #measure
    
elif computer == 'setup': # infos @ desktop --> right click --> nvidia control panel
    refresh_rate = 120 # checked, correct
    mon_size = np.array([1920, 1080]) # checked, correct
    mon_height = 29.1 # checked, correct
    mon_width = 52.2 # checked, correct
    mon_distance = 62 # checked, correct
    
elif computer == 'MPI':
    refresh_rate = 60 # look up
    mon_size = np.array([2560, 1440]) # look up
    mon_height = 17.43 # look up
    mon_width = 30 # checked, correct (in cm)
    mon_distance = 55 # checked, correct (in cm)
    
#Timing (in s)
frame = 1/refresh_rate # time per frame

time_cue = 0.5/frame
time_templ = 4/frame 
time_gabor = 0.2/frame 
time_mask = 0.2/frame 
time_del = 1/frame #1
time_probe = 10/frame
time_iti = (np.linspace(0.5, 0.9, num=6))
time_feedback = 2/frame
time_countdown = 1/frame

time_templ_angletraining = 1/frame
time_gabor_angletraining = 1/frame
time_mask_angletraining = 0.5/frame

time_all = 3/frame


#Trigger codes (for eyetracking)
triggers_cue = [1, 2] # 1 = Template 1 active, 2 = Template 2 active
triggers_miniblock = [70,71,75] # 70 = Miniblock start (first reproduction), 71 = Miniblock end (second reproduction), 75 = neither (middle trial)
triggers_gabor = np.arange(10, 10+n_angle_bins) #corresponds to the individual orientation bins
triggers_template1 = np.arange(50,50+n_template1_bins)
triggers_template2 = np.arange(50+n_template1_bins,50+n_template1_bins+n_template2_bins)
triggers_probe = [0,1] # 0 = memory not probed, 1 = memory probed
triggers_break = [90,91] # 0 = no break after trial, 1 = break after trial
triggers_pause = [92,93]

 
##Screen parameters
screen_color = 105 #'DimGrey'

##Text parameters
text_color = 'White'
if computer != 'setup':
    text_size = 0.25 #in degrees of visual angle
else:
    text_size = 0.5
text_font = 'Arial'

##Fixation parameters
if computer != 'setup':
    fix_size = 0.1 #in degrees of visual angle
else:
    fix_size = 0.3
fix_lineWidth = 3
fix_lineColor = 40
fix_value = 0 # for black in hsv


##Cue parameters
cue_size = 0.6
cue_units = 'deg'

##Gabor parameters
gabor_tex = 'sin'
gabor_mask = 'gauss'
gabor_units = 'deg'
if computer != 'setup':
    gabor_sf = 2.4 #spatial frequency of Gabor (Fritsche et al., 2017: 0.33; Pascucci et al., 2021)
else:
    gabor_sf = 1.5
gabor_phase = 0.5
gabor_contrast = 0.3 #0.25 should, technically, be the equivalent of a 25% Michelson contrast
#gabor_contrast = 0.5 # used in first pilot 08/11/21 bc 0.25 was barely visible
gabor_maskSD = None#{'sd': 1} # same parameters as in Fritsche
if computer != 'setup':
    gabor_size = [1.5, 1.5]
else:
    gabor_size = [3, 3] # originally [2, 2], used to make stimulus appear same size as templates

##Mask parameters
mask_noiseType = 'white'
mask_mask = 'gauss'
mask_units = 'deg'
mask_noiseElementSize = 5 # 1 in Darinka's script
mask_size = gabor_size
mask_contrast = 3 # not a part of Darinka's script

##Response Bar parameters
respBar_width = 0.5 #0.2 in Darinka's script
if computer != 'setup':
    respBar_height = 1.5
else:
    respBar_height = 4.5 # 2 in Darinka's script
respBar_units = 'deg'
respBar_lineColor = 'White'
respBar_fillColor = 'White'

respBar_color = 0 # hue of hsv-responsebar (made by the create_templbar function)
respBar_saturation = 0 # saturation of hsv-responsebar (made by the create_templbar function)

##Template parameters (for both circle and line)
#color parameters
templ_colorspace = 'hsv'
templ_value = .8 # color value in hsv space --> not 1 = not as "bright", "more black"

#size parameters
templ_width = 0.15 # in Muhle-Karbe, Stokes 2021: 0.25
if computer != 'setup':
    templ_height = 2
else:
    templ_height = 2 # as in Muhle-Karbe, Stokes 2021
templ_units = 'deg'

### Instructions ###
if language == 'English':
    run_eyetrackerSetUp = 'Press <E> to run the eyetracker again, or press <C> to run the experiment.'
    

    inst_beginning = 'Welcome!\n\n '\
                'In this experiment, you will run through multiple blocks of tasks. \n '\
                'In every block, you will be shown two new and different oriented bars.\n\n '\
                'Your primary task will be to remember those orientations.\n '\
                'Then, you will be asked to reproduce those orientations.\n'\
                'You will do this by turning a white bar until its orientation matches the one you just saw,\n '\
                "Now don't forget those orientations! You will be asked to reproduce them again at the end of the block.\n"\
                'While you remember your orientations, one patch of lines after another will be shown to you. \n'\
                'Your task is to reproduce the orientation of those patches the same way you did in the beginning. \n'\
                "You don't have to remember the patch orientations! Only remember the orientations of the bars shown to you at the start of every block. \n"\
                'After reproducing the patch, please indicate whether the patch was oriented \n' \
                'In clockwise (CW) or counterclockwise (CCW) direction toward the bar that was cued to you before the patch appeared \n '\
                'Remember to always consider the smallest possible angle between patch and bar for this task. \n\n'\
                'Please remember to respond as fast and accurately as possible \n'\
                'and to always keep your eyes on the fixation cross in the middle of the screen.\n\n '\
                'Thanks for your participation!\n\n '\
                'Press the <confirmation button> to continue.'

    ready_5 = 'Ready?\n\n '\
              'The experiment will start in 5 seconds.' 
    ready_4 = 'Ready?\n\n '\
              'The experiment will start in 4 seconds.' 
    ready_3 = 'Ready?\n\n '\
              'The experiment will start in 3 seconds.' 
    ready_2 = 'Ready?\n\n '\
              'The experiment will start in 2 seconds.' 
    ready_1 = 'Ready?\n\n '\
              'The experiment will start in 1 second.' 
    
    inst_templates_1 = 'Remember'
                                
    inst_templates_2 = 'Reproduce Lines'
    
    inst_templates_3 = 'Reproduce Lines'
    
    inst_reproduction = 'Reproduce patches'
    
    exp_break_short = 'You may take a short break.\n\n '\
                'Please leave your head on the chinrest. \n\n '\
                'Press the <confirmation button> when you are ready to continue.'
                
    exp_break_long = 'You may take a break. \n\n '\
                    'If you want, you can get up and stretch. We will re-calibrate your eyes when you are done. \n\n '\
                     'Press the <confirmation button> when you are ready to continue.'   
    
    memory_probe_text_1 = 'CCW \n\n '\
                        '----------------------------- \n\n '\
                            'CW'
                            
    memory_probe_text_2 = 'CW \n\n '\
                        '----------------------------- \n\n '\
                            'CCW \n\n '
                                         
    angletraining_text = 'This session is to train you on reporting the correct (smaller) angle between two orientations. \n\n ' \
                        'Please look at the two oriented lines. \n\n '\
                        'Then report whether the red bar is oriented clockwise (CW) or counterclockwise (CCW) with respect to the blue bar. \n\n '\
                        'Your aim should be to correctly respond to 90% of trials. \n\n '\
                        'Have fun! \n\n'\
                        'Press the <confirmation button> to continue.'
    
    angletraining_goodbye = "Let's see your scores..."
    
    angletraining_feedback_1 = "You got "
    
    angletraining_feedback_2 = "% of trials correct. \n\n "\
                            "Press the <confirmation button> to continue to the break screen."
                            

    goodbye = 'You have reached the end of your session.\n\n '\
              'Thanks a lot for having participated!'
    
    goodbye_training = 'You finished your training blocks. \n\n'\
                    'Please come outside.'
    
    between_passes = 'Please do not confirm immediately. We will have a talk before we continue. ' 

if language == 'German':
    run_eyetrackerSetUp = 'Press <E> to run the eyetracker again, or press <C> to run the experiment.'
    
    inst_beginning = 'Hallo! \n\n '\
            'Dieses Experiment ist in Blocks unterteilt. \n' \
            'In jedem Block werden dir zwei neue, unterschiedliche gekippte Balken gezeigt. \n\n '\
            'Deine erste Aufgabe ist es, dir deren Orientierungen zu merken. \n '\
            'Dann wirst du gebeten, diese Orientierungen wiederzugeben. \n '\
            'Das tust du, indem du einen weißen Balken drehst, bis er die Orientierung wiedergibt, die du gesehen hast. \n '\
            'Jetzt die Orientierungen nicht wieder vergessen! Am Ende des Blocks wirst du sie noch einmal wiedergeben müssen. \n '\
            'Während du die Orientierungen im Kopf behältst, wird dir ein Linienmuster nach dem anderen gezeigt. \n'\
            'Deine Aufgabe ist es jetzt, die Orientierungen dieser Muster wiederzugeben, so wie die der Balken am Anfang. \n '\
            'Die Orientierung der Muster musst du dir nicht merken! Nur die der Balken am Anfang jedes Blocks. \n '\
            'Nachdem du die Orientierung eines Musters wiedergegeben hast, gib bitte an, ob das Muster im Vergleich zum angegebenen Balken \n '\
            'Im Uhrzeigersinn (CW) oder gegen den Uhrzeigersinn (CCW) gekippt war. \n'\
            'Denk daran, dabei immer auf den kleinstmöglichen Winkel zwischen Muster und Balken zu achten. \n\n'\
            'Bitte antworte immer so schnell und genau wie du kannst, \n'\
            'und denke daran, mit den Augen immer in die Mitte des Bildschirms (auf das Kreuz) zu schauen. \n\n '\
            'Danke für deine Teilnahme! \n\n'\
            'Drück auf die <Eingabetaste>, um fortzufahren.'

    ready_5 = 'Bereit?\n\n '\
              'Das Experiment beginnt in 5 Sekunden.' 
    ready_4 = 'Bereit?\n\n '\
              'Das Experiment beginnt in 4 Sekunden.' 
    ready_3 = 'Bereit?\n\n '\
              'Das Experiment beginnt in 3 Sekunden.' 
    ready_2 = 'Bereit?\n\n '\
              'Das Experiment beginnt in 2 Sekunden.' 
    ready_1 = 'Bereit?\n\n '\
              'Das Experiment beginnt in 1 Sekunde.'  
    
    inst_templates_1 = 'Merken'
                                
    inst_templates_2 = 'Balken wiedergeben'
    
    inst_templates_3 = 'Balken wiedergeben.'
    
    inst_reproduction = 'Muster wiedergeben.'
    
    exp_break_short = 'Du darfst kurz Pause machen.\n\n '\
                'Bitte lass deinen Kopf auf der Kinnstütze. \n\n '\
                'Drück die <Eingabetaste>, wenn du weitermachen möchtest.'
                
    exp_break_long = 'Du darfst eine Pause machen. \n\n '\
                     'Wenn du möchtest, kannst du aufstehen und dich strecken. Wir werden deine Augen re-kalibrieren, wenn du bereit bist. \n\n '\
                    'Drück die <Eingabetaste>, wenn du weitermachen möchtest.'   
    
    memory_probe_text_1 = 'CCW \n\n '\
                        '----------------------------- \n\n '\
                            'CW'
                            
    memory_probe_text_2 = 'CW \n\n '\
                        '----------------------------- \n\n '\
                            'CCW \n\n '
               
    angletraining_text = 'In diesem Abschnitt übst du, den richtigen (kleineren) Winkel zwischen zwei Orientierungen richtig zu benennen. \n\n ' \
                        'Bitte schau dir die zwei gekippten Balken an. \n\n '\
                        'Dann gib an, ob der rote Balken im Vergleich zum blauen Balken im Uhrzeigersinn (CW) oder gegen den Uhrzeigersinn (CCW) gekippt ist. \n\n '\
                        'Das Ziel ist, 90% der Versuche richtig zu beantworten. \n\n '\
                        'Viel Spaß! \n\n'\
                        'Drück auf die <Eingabetaste>, um fortzufahren.'
    
    angletraining_goodbye = "Sehen wir uns deine Ergebnisse an..."
    
    angletraining_feedback_1 = "Du hast "
    
    angletraining_feedback_2 = "% der Versuche richtig beantwortet. \n\n "\
                            "Drück auf die <Eingabetaste>, um zum Pausenbildschirm fortzufahren."
                            
    goodbye = 'Du hast das Ende des heutigen Experiments erreicht.\n\n '\
              'Vielen Dank für deine Teilnahme!'
    
    goodbye_training = 'Du hast deine Trainings-Blocks geschafft. \n\n'\
                    'Bitte komm nach draußen.'