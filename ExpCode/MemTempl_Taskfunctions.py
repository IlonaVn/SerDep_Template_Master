# -*- coding: utf-8 -*-

#Purpose: This file contains basic functions needed to run an experiment
#Author: Ilona Vieten (adapted from Darinka Truebutschek)
#Date created: 06/10/2021
#Date last modified: 08/04/2022
#Python version: 3.6.13

import numpy as np

from psychopy import core, event, visual, logging
from psychopy.hardware import keyboard
from MemTempl_config import path_main

'''
====================================================
Functions needed for feedback
====================================================
'''

def angletraining_ratio_correct(df,last_trial,rating_range,window=10):
    """
    Function to determine if participant made correct ("1") or incorrect ("0") cwccw judgments
    by comparing answers with predetermined correct answers from trialsequence (from determine_correct_answers_cwccw()).
    Returns ratio of correct answers in % of counted trials.
    
    Input dataframe is always longer than number of trials that are actually performed.
    
    :param df: complete trialsequence dataframe for one subject
    :param last_trial: last trial that was performed out of whole dataframe
    :param rating_range: range of distances between templates (only whole numbers!) that are considered.
                        Trials with distances not in the array will be excluded from ratio (because too close to 0°/90°).
    :param window: over how many trials before the last is ratio determined
    
    """
    #select prepared trials that were actually performed
    df_sel = df[:last_trial]
    
    #select those trials that are considered for performance rating
    correct_sel = df_sel.loc[df_sel['angles'].isin(rating_range)].reset_index(drop=True)
    
    #select latest trials (number specified by param window)
    if len(correct_sel) >= window:
        print('last trial considered:' + str(len(correct_sel)))
        sel = correct_sel[(len(correct_sel)-window):len(correct_sel)].reset_index(drop=True)
    
    else:
        print('window too large/initial block too short')
        sel = correct_sel.copy(deep=True)
    
    #establish variable in which the number of correct answers is counted
    cwccw_correct = np.zeros((len(sel)))
    for i in range(len(sel)):
        if sel.loc[i,'correct_answer'] == sel.loc[i,'cwccw_resp']:
            cwccw_correct[i] = 1
    
    #determine ratio of correct answers
    ratio = (np.count_nonzero(cwccw_correct == 1)/len(cwccw_correct))*100
    ratio = np.round(ratio, 1)
    
    return ratio

def experiment_ratio_correct(df):
    """
    Function to determine if participant made correct ("1") or incorrect ("0") cwccw judgments
    by comparing answers with predetermined correct answers from trialsequence (from determine_correct_answers_cwccw()).
    Returns ratio of correct answers in % of counted trials.
    
    :param df: complete or part of trialsequence dataframe for one subject
    
    """
    #drop trials that do not have a correct answer (when distance is 0 or 90 degrees)
    sel = df[df['correct_cwccw'] != 2].reset_index(drop=True) 
    cwccw_correct = np.zeros((len(sel)))
    
    #For each trial, determine if correct response was given
    for i in range(len(sel)):
        if sel.loc[i,'correct_cwccw'] == sel.loc[i,'cwccw_resp']:
            cwccw_correct[i] = 1
    
    #Determine ratio of correct responses over all trials
    ratio = (np.count_nonzero(cwccw_correct == 1)/len(cwccw_correct))*100
    ratio = np.round(ratio, 1)
    
    return ratio

'''
====================================================
Functions needed for trial sequence generation
====================================================
'''

def determine_correct_answers_cwccw(df, angletraining = 0):
    """
    Function to determine the correct answer in cwccw task or angletraining (before participant input).
    Note: response of "0" is always counterclockwise, and response of "1" is always clockwise.
        This does not change with button assignment.
    
    :param df: trialsequence dataframe for one subject
    :param angletraining: 1 = determination for angletraining, not main task
    
    """
    from MemTempl_Taskfunctions import direction_extent_of_angle_error_cwccw
    
    cwccw_correct = np.zeros((len(df),1))
    
    if angletraining == 1:
        template = 'template_1'
        item = 'template_2'
    
    #for each trial, determine stimuli orientations to be compared
    for i in range(len(df)):
        if angletraining == 0:
            item = 'MemItem_angle'
            if df.loc[i, 'Active'] == 1:
                template = 'MemTempl1_angle'
            elif df.loc[i, 'Active'] == 2:
                template = 'MemTempl2_angle'
            
        temp = direction_extent_of_angle_error_cwccw(df.loc[i, template],df.loc[i,item])

        if temp == 90 or temp == 0:
            cwccw_correct[i] = 2 # no correct response possible
            
        if temp < 0: #counterclockwise as determined by direction_extent funtion (difference between 0 and 90 or under -90)
            cwccw_correct[i] = 0

        if temp > 0: #clockwise as determined by direction_extent function (difference between 0 and -90 or over 90)
            cwccw_correct[i] = 1
        
    return cwccw_correct

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

'''
====================================================
Basic experimental components
====================================================
'''

def create_text(win, text, size, font, color):
    """
    Function to create text component.
    
    :param win: which window to present text in
    :param text: text to present
    :param size: size of text
    :param font: font to be used
    :param color: color of text

    """
    textStim = visual.TextStim(win, text=text, units='deg', height=size, font=font, color=color, wrapWidth=100)

    return textStim

def create_fix(win, size, lineWidth, lineColor, value, colorspace):
    """
    Function to create fixation cross component.
    
    :param win: which window to present fixation in
    :param size: size of fixation cross
    :param lineWidth: line width of fixation cross
    :param lineColor: color of fixation cross (hue for hsv color space)
    :param value: value for hsv color space
    :param colorspace: needs to be 'hsv' as color is input as 3-item list (color_fix)
    
    """
    
    #create hsv color - saturation is always = 1
    color_fix = [lineColor,1,value]    
        
    #create fixation cross
    fixStim = visual.ShapeStim(win, units='deg', vertices=((0, -size), (0, size), (0,0), (-size, 0), (size, 0)), lineWidth=lineWidth, closeShape=False, colorSpace=colorspace, lineColor=color_fix)

    return fixStim

def create_Gabor(win, tex, mask, units, sf, phase, contrast, maskParams, orient, size):
    """
    Function to create Gabor stimulus component.
    
    :param win: which window to present Gabor in
    :param tex: which texture to be used on the stimulus (i.e., Gabor=sin)
    :param mask: which alpha mask to use, determining the shape
    :param units: units to use when drawing
    :param sf: spatial frequency of the Gabor
    :param phase: phase of the Gabor (i.e., where are the black lines with respect to the white lines)
    :param contrast: contrast of the mask 
    :param maskParams: sd of mask
    :param orient: which orientation Gabor should be displayed in
    :param size: size of Gabor

    """
    gaborStim = visual.GratingStim(win, tex=tex, mask=mask, units=units, sf=sf, phase=phase, contrast=contrast, maskParams=maskParams, ori=orient, size=size)

    return gaborStim

def create_Mask(win, noiseType, mask, units, noiseElementSize, size, contrast):
    """
    Function to create mask stimulus component.
    
    :param win: which window to present mask in
    :param noiseTyp: which type of noise to apply (e.g., white noise)
    :param mask: which alpha mask to use, determining the shape
    :param units: units to use when drawing
    :param noiseElementSize: size of individual noise elements 
    :param size: size of mask
    :param contrast: contrast of mask

    """
    from scipy.ndimage import gaussian_filter

    maskStim = visual.NoiseStim(win, noiseType=noiseType, mask=mask, units=units, noiseElementSize=noiseElementSize, size=size, contrast = contrast)

    #Smooth stimulus with Gaussian kernel
    maskStim.tex = gaussian_filter(maskStim.tex, sigma=2)
    
    return maskStim

def create_respBar(win, bg_color, width, height, units, orient):
    """
    Function to create response bar.
    
    :param win: which window to present the response bar in
    :param bg_color: background color of the screen
    :param width: width of the response bar
    :param height: height of the response bar
    :param units: units to use when drawing
    :param orient: which orientation the response bar should be displayed in

    """
    from scipy.ndimage import gaussian_filter
    from psychopy.tools.monitorunittools import deg2pix

    #First, define size of stimulus in pixels
    barSize = np.round(deg2pix(height, win.monitor))
    barWidth = np.round(deg2pix(width, win.monitor))

    respBar = np.ones([256, 256, 3])*(bg_color/255) 
    respBar[int(np.shape(respBar)[0]/2 - barSize/2) : int(np.shape(respBar)[0]/2 + barSize/2), int(np.shape(respBar)[0]/2 - barWidth/2) : int(np.shape(respBar)[0]/2 + barWidth/2), 0] = 1
    respBar[int(np.shape(respBar)[0]/2 - barSize/2) : int(np.shape(respBar)[0]/2 + barSize/2), int(np.shape(respBar)[0]/2 - barWidth/2) : int(np.shape(respBar)[0]/2 + barWidth/2), 1] = 1
    respBar[int(np.shape(respBar)[0]/2 - barSize/2) : int(np.shape(respBar)[0]/2 + barSize/2), int(np.shape(respBar)[0]/2 - barWidth/2) : int(np.shape(respBar)[0]/2 + barWidth/2), 2] = 1

    respBar = gaussian_filter(respBar, sigma=[20, 3, 0])

    respBarStim = visual.ImageStim(win, image=respBar, units='deg', ori=orient, size=2)

    return respBarStim

def create_cue(win, height, units, colorspace, color, value):
    """
    Function to create color cue component.
    
    :param win: which window to present the color cue in
    :param height: height of the color cue
    :param units: units to use when drawing
    :param colorspace: needs to be 'hsv' as color is input as 3-item list (hsv_color)
    :param color: hue for hsv color space color
    :param value: value for hsv color space color

    """

    from psychopy.tools.monitorunittools import deg2pix

    #First, define size of stimulus in pixels
    circleSize = np.round(deg2pix(height, win.monitor))
    
    #Define width of circle line in pixels
    circleWidth = 1 #one pixel wide
    
    #only hue is pulled from trial sequence, the rest of the color has to be filled in here
    hsv_color = [color,1,value]

    #Then, draw circle with that radius and that width
    cueCircleStim = visual.Circle(win,units="pix", radius = circleSize/2, edges = 150, lineWidth = circleWidth,  colorSpace = colorspace, fillColor = hsv_color, lineColor = hsv_color)#, borderColor_rgb = [0.5,0.5,0.5]

    return cueCircleStim

def create_templcircle(win, width, height, units, colorspace, color, value, saturation):
    """
    Function to create circle component of template stimulus.
    
    :param win: which window to present the template in
    :param width: width of the template lines
    :param height: height of the template circle
    :param units: units to use when drawing
    :param colorspace: needs to be 'hsv' as color is input as 3-item list (hsv_color)
    :param color: hue for hsv color space color
    :param value: value for hsv color space color
    :param saturation: value for hsv color space color

    """
    from psychopy.tools.monitorunittools import deg2pix

    #First, define size of stimulus in pixels
    circleSize = np.round(deg2pix(height, win.monitor))
    
    #Define width of circle line in pixels
    circleWidth = np.round(deg2pix(width, win.monitor))
    
    #only hue is pulled from trial sequence, the rest of the color has to be filled in here
    hsv_color = [color,saturation,value]

    #Then, draw circle with that radius and that width
    templrCircleStim = visual.Circle(win,units="pix", radius = circleSize/2, edges = 150, lineWidth = circleWidth,  colorSpace = colorspace, fillColor = None, lineColor = hsv_color)

    return templrCircleStim

def create_templbar(win, width, height, units, orient, colorspace, color, value, saturation):
    """
    Function to create bar component of template stimulus.
    
    :param win: which window to present the template in
    :param width: width of the template lines
    :param height: height of the template circle/bar
    :param units: units to use when drawing
    :param orient: which orientation the response bar should be displayed in
    :param colorspace: needs to be 'hsv' as color is input as 3-item list (hsv_color)
    :param color: hue for hsv color space color
    :param value: value for hsv color space color
    :param saturation: value for hsv color space color

    """

    from psychopy.tools.monitorunittools import deg2pix

    #Define start and end points of line
    barHeight = np.round(deg2pix(height, win.monitor))
    start_point = ((0,-(barHeight/2)))
    end_point = ((0,barHeight/2))
    
    #Define width of line
    barWidth = np.round(deg2pix(width, win.monitor))
    
    #only hue is pulled from trial sequence, the rest of the color has to be filled in here
    hsv_color = [color,saturation,value]

    templBarStim = visual.Line(win, units = 'pix', ori=orient,start = start_point, end = end_point, lineWidth = barWidth, colorSpace = colorspace, lineColor = hsv_color)
    return templBarStim

'''
====================================================
Functions to run an experiment
====================================================
'''    

def run_text(win, text, size, font, color, duration, button):
    """
    Function to run text stimulus.
    
    :param win: which window to present text in
    :param text: text to present
    :param size: size of text
    :param font: font to be used
    :param color: color of text
    :param duration: 'inf' = wait for pre-specified button press, else wait for pre-specified time (in frames)
    :param button: if duration is "inf", button to continue to next screen

    """
    from MemTempl_Taskfunctions import create_text

    textStim = create_text(win, text, size, font, color)
    
    if duration == 'inf':
        textStim.draw()
        win.flip()
        event.waitKeys(keyList = button)
    else:
        for framei in np.arange(duration): #duration defined as frames not seconds
            if 0 <= framei <= duration:
                textStim.draw()
            
            win.flip()

def run_fix(win, size, lineWidth, lineColor, duration, fix_value, colorspace):
    """
    Function to run fixation cross stimulus.

    :param win: which window to present fixation in
    :param size: size of fixation cross
    :param lineWidth: lineWidth of fixation cross
    :param linecolor: color of fixation cross (hsv hue)
    :param duration: wait for pre-specified time (in frames)
    :param fix_value: color of fixation cross (hsv value)
    :param colorspace: needs to be 'hsv' as color is input as 3-item list to create_fix

    """
    from MemTempl_Taskfunctions import create_fix

    fixStim = create_fix(win, size, lineWidth, lineColor, fix_value, colorspace)
    fixStim.draw()
    win.flip()
    core.wait(duration)
    
def run_stim(win, stim, stim2, stim3, duration, log_name, button = 'return'):
    """
    Function to run stimulus with specified duration or manual termination.
    
    :param win: which window to present text in
    :param stim: which stimulus to present
    :param stim2: additional stimulus to present super-imposed
    :param stim3: additional stimulus to present super-imposed
    :param duration: wait for pre-specified time (in frames/second)
    :param log_name: name of stimulus to be added to logfile
    :param button: button to press to move to next screen if duration is = 'inf'

    """
    if duration == 'inf':
        stim.draw()
        win.flip()
        event.waitKeys(keyList = button)
        
    else:
        for framei in np.arange(duration): #duration defined as frames not seconds
            if 0 <= framei <= duration:
                stim.draw()
        
                if stim2 != False:
                    stim2.draw()
                    
                if stim3 != False:
                    stim3.draw()
                    
            win.logOnFlip(level=logging.EXP, msg=log_name)
            win.flip()
            if framei == 1:
                win.getMovieFrame()
                win.saveMovieFrames(fileName=str(path_main) + '/' + 'Stimuli' + '/' + log_name + '.tiff', clearFrames=True)

def update_stim(win, stim, stim2, updateKeys, logKeys):
    """
    Function to run continuously changing stimulus (= rotating response bar)
    and record responses.
    
    :param win: which  window to present text in
    :param stim: which stimulus to present
    :param stim2: second stimulus to present superimposed
    :param updateKeys: which keyboard keys will update the stimulus
    :param logKeys: which keyboard key will register the response
    :return: last button press & angle of stimulus

    """
    keepGoing = True
    allKeys_name = []  #dummy variable to keep track of all keys having been pressed
    allKeys_rt = []
    allKeys_duration = []

    #Initialize important variables
    alpha = stim.ori #original orientation of the response bar

    #Initialize keyboard 
    kb = keyboard.Keyboard()

    #Start listening to keyboard
    win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
    win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
    
    kb.start()
    while keepGoing:
        keys = kb.getKeys(waitRelease=False, clear=False)

        for key in keys:
            allKeys_name.append(key.name)  # store all recorded button presses
            allKeys_rt.append(key.rt)
            allKeys_duration.append(key.duration)

        stim.draw()
        win.flip()
        
        #Exit loop when response has been commited
        if logKeys in keys:
            keepGoing = False
            if stim.ori > 180:
                alpha = stim.ori-180
            elif stim.ori < 0 and stim.ori >= -180:
                alpha = stim.ori+180
            elif stim.ori < -180:
                alpha = stim.ori+360
            else:    
                alpha = stim.ori
            kb.stop() # "stopping key buffers" warning comes from this function

        if len(keys) > 0:
            while (updateKeys[0] in keys): #clockwise
                stim.ori += 0.5 #turn back to 1 maybe

                #To always reset to circular response
                if stim.ori > 360:
                    stim.ori = np.mod(stim.ori, 360)

                keys = kb.getKeys(clear=True) #stops rotation at key release

                for key in keys:
                    allKeys_name.append(key.name) #store all recorded button presses
                    allKeys_rt.append(key.rt)
                    allKeys_duration.append(key.duration)
            while (updateKeys[1] in keys): #counter-clockwise
                stim.ori -= 0.5

                #To always reset to circular response
                if np.abs(stim.ori) > 360:
                    stim.ori = np.mod(stim.ori, 360)

                keys = kb.getKeys(clear=True) #stops rotation at key release

                for key in keys:
                    allKeys_name.append(key.name) #store all recorded button presses
                    allKeys_rt.append(key.rt)
                    allKeys_duration.append(key.duration)
                    
    return allKeys_name, allKeys_rt, allKeys_duration, alpha

def cwccw_stim(win, stim, stim2, keyscwccw):
    """
    Function to run judgment task screen and record response.
    
    :param win: which  window to present text in
    :param stim: which stimulus to present
    :param stim2: second stimulus to present superimposed
    :param keyscwccw: which key denotes which response

    """
    keepGoing = True
    allKeys_name = []  #dummy variable to keep track of all keys having been pressed
    allKeys_rt = []
    allKeys_duration = []

    #Initialize keyboard 
    kb = keyboard.Keyboard()

    #Start listening to keyboard
    win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
    win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
    
    kb.start()
    
    while keepGoing:
        keys = kb.getKeys(waitRelease=False, clear=False)

        for key in keys:
            allKeys_name.append(key.name)  # store all recorded button presses
            allKeys_rt.append(key.rt)
            allKeys_duration.append(key.duration)

        stim.draw()
        win.flip()

        #Exit loop when response has been commited
        if keyscwccw[0] in keys:
            keepGoing = False
            response = 0
            rt = key.rt
            kb.stop() # "stopping key buffers" warning comes from this function
        
        elif keyscwccw[1] in keys:
            keepGoing = False
            response = 1
            rt = key.rt
            kb.stop() # "stopping key buffers" warning comes from this function

    return response, rt

def run_angletraining(win, start_at_trial, length_initial_block, length_additional_block, df = None):
    """
    Function to run specified number of trials from pre-made trial sequence.
    
    :param win: window to use to display stimuli
    :param start_at_trial: index number of first trial in run (either 0 or something beyond length_initial_block)
    :param length_initial_block: number of trials in first run
    :param length_additional_block: number of trials added in subsequent run if threshold is not reached in first run
    :param df: dataframe containing angletraining trial sequence
    """
    from MemTempl_Taskfunctions import run_stim, run_text, create_templcircle, create_templbar, create_text
    from MemTempl_config import angletraining_text, text_size, text_font, text_color, fix_size, fix_lineWidth, fix_lineColor, fix_value, templ_colorspace, templ_width, templ_height, templ_units, templ_value, time_del, time_templ_angletraining, memory_probe_text_1, memory_probe_text_2, keysup, keysdown, cue_size, cue_units, time_feedback
    from MemTempl_trialsequence import training_prepTrials
    
    #set colors (hsv color space hues)
    white_color = 0 #white template color
    blue_color = 227 #blue template color
    green_color = 124 #green feedback color
    red_color = 4 #red feedback color/template color
    
    #initialize cwccwkeys
    cwccwkeys = [0,0]

    #generate trials
    if start_at_trial == 0:
        df_training = training_prepTrials()
        until_trial = length_initial_block
    else:
        df_training = df
        until_trial = start_at_trial + length_additional_block
    
    #welcome screen
    if start_at_trial == 0:
        run_text(win, angletraining_text, text_size, text_font, text_color, duration='inf', button='return')
    
    #iterate through specified number of trials
    for triali in range(start_at_trial, until_trial):
        
        #extract trial information
        templ_orient = df_training.loc[triali,'template_1']
        gabor_orient = df_training.loc[triali,'template_2']
        
        if df_training.loc[triali, 'ccw_up'] == 1:
            probe_text = memory_probe_text_1
            cwccwkeys[0] = keysup
            cwccwkeys[1] = keysdown
        else:
            probe_text = memory_probe_text_2
            cwccwkeys[0] = keysdown
            cwccwkeys[1] = keysup
            
        #create stimuli
        stim_fix = create_fix(win, fix_size, fix_lineWidth, fix_lineColor, fix_value, templ_colorspace)
        stim_feedbackcircle = create_templcircle(win, templ_width, templ_height, templ_units, templ_colorspace, white_color, templ_value, saturation = 0)
        stim_templ1circle = create_templcircle(win, templ_width, templ_height, templ_units, templ_colorspace, blue_color, templ_value, saturation = 1)
        stim_templ1bar = create_templbar(win, templ_width, templ_height, templ_units, templ_orient, templ_colorspace, blue_color, templ_value, saturation = 1)
        stim_templ2circle = create_templcircle(win, templ_width, templ_height, templ_units, templ_colorspace, red_color, templ_value, saturation = 1)
        stim_templ2bar = create_templbar(win, templ_width, templ_height, templ_units, gabor_orient, templ_colorspace, red_color, templ_value, saturation = 1)        
        stim_probe_text = create_text(win, probe_text, text_size, text_font, text_color)
        stim_green_feedb = create_cue(win, cue_size, cue_units, templ_colorspace, green_color, templ_value)
        stim_red_feedb = create_cue(win, cue_size, cue_units, templ_colorspace, red_color, templ_value)
        
        ##run trial sequence and record responses
        #Clear all keyboard events from previous trial
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
        
        #run both template-shape stimuli
        run_stim(win, stim=stim_fix, stim2=False, stim3 = False, duration=time_del, log_name='Delay')
        run_stim(win, stim=stim_templ1bar,stim2=stim_templ1circle, stim3 = False,duration=time_templ_angletraining,log_name='Templ')
        run_stim(win, stim=stim_fix, stim2=False,stim3 = False,duration=time_del, log_name='Delay')
        run_stim(win, stim=stim_templ2bar,stim2=stim_templ2circle, stim3 = False,duration=time_templ_angletraining,log_name='Templ')
        run_stim(win, stim=stim_fix, stim2=False, stim3 = False,duration=time_del, log_name='Delay')
        
        #run window asking for response
        cwccw_resp, cwccw_rt = cwccw_stim(win, stim=stim_probe_text, stim2=False, keyscwccw = cwccwkeys) # response screen
        
        #run window providing feedback
        if cwccw_resp == df_training.at[triali,'correct_answer']:
            run_stim(win, stim=stim_green_feedb, stim2=False, stim3 = False,duration=time_del, log_name='Feedback')
        else:
            run_stim(win, stim=stim_red_feedb, stim2=False, stim3 = False,duration=time_del, log_name='Feedback')
        
        #run an overlay of both template-shape stimuli as feedback
        run_stim(win,stim=stim_templ1bar,stim2=stim_templ2bar, stim3 = stim_feedbackcircle,duration=time_feedback,log_name='Templ')
        
        #save cwccw-task response and reaction time to dataframe
        df_training.at[triali,'cwccw_resp'] = cwccw_resp
        df_training.at[triali,'cwccw_RT'] = cwccw_rt
        
    return df_training
    

def run_trial(win, bg_color, break_trial, pause_trial, break_text_short, break_text_long, text_size, text_color, text_font, memory_probe_text_1, memory_probe_text_2, 
    trial_active, trial_ccw_up, inst_reproduction, templ_instr_text_1, templ_instr_text_2, post_templ_text, 
    cue_size, cue_units, templ1_cue_first_repr1, templ1_cue_first_repr2,
    gabor_tex, gabor_mask, gabor_units, gabor_sf, gabor_phase, gabor_contrast, gabor_maskParams, gabor_orient, gabor_size, gabor_duration,
    mask_noiseType, mask_mask, mask_units, mask_noiseElementSize, mask_size, mask_duration, mask_contrast,
    fix_size, fix_lineWidth, fix_lineColor, fix_duration, fix_value,
    resp_width, resp_height, resp_units, respBarpool, respBar_color, resp_updateKeys, resp_logKeys, keyscwccw,
    iti_duration,
    eyetracking, tracker,
    triggersCue, triggersGabor, triggersMiniblock, triggersProbe, triggersTempl1, triggersTempl2, triggersBreak, triggersPause,
    rem_templ, end_rememb, templ_width, templ_height, templ_units, templ_duration, templ1_orient, templ2_orient, templ_colorspace, templ1_color, templ2_color, templ_value,
    cwccw_task, trialSeq, triali, feedback_1, feedback_2, last_break_triali, rawDat_filename, training):
    """
    Function to run one trial of main task.

    :param win: which window to present trial in - defined in Main
    :param bg_color: background color of the screen - defined in configuration
    
    :param break_trial: bool, whether trial is followed by short (non-moving) break for subject - defined in Main
    :param pause_trial: bool, whether trial is followed by long (movement allowed) break for subject - defined in Main
    
    :param break_text_short: text announcing short break to subject - defined in configuration (exp_break_short)
    :param break_text_long: text announcing long break to subject - defined in configuration (exp_break_long)
    :param text_size: text size - defined in configuration
    :param text_color: text color - defined in configuration
    :param text_font: text font - defined in configuration
    :param memory_probe_text_1: judgment task screen, CCW on top - defined in configuration
    :param memory_probe_text_2: judgment task screen, CW on top - defined in configuration

    :param trial_active: 1 or 2, which template is active in current trial - defined in Main
    :param trial_ccw_up: 1 if ccw is to be displayed on top (i.e., if memory_probe_text_1 is to be displayed), 0 if cw is - defined in Main
    
    :param inst_reproduction: short instruction for reproduction task - defined in configuration
    :param templ_instr_text_1: short instruction cuing to remember template - defined in configuration (inst_templates_1)
    :param templ_instr_text_2: short instruction cuing to reproduce template first time - defined in configuration (inst_templates_2)
    :param post_templ_text: short instruction cuing to reproduce template second time - defined in configuration (inst_templates_3)
    
    :param cue_size: size of cue - defined in configuration
    :param cue_units: units of cue size - defined in configuration

    :param templ1_cue_first_repr1: bool, whether template 1 is cued first in first reproduction - defined in Main (templ1_first_repr1)
    :param templ1_cue_first_repr2: bool, whether template 1 is cued first in second reproduction - defined in Main (templ1_first_repr2)

    :param gabor_tex: which texture to be used on the stimulus (i.e., Gabor=sin) - defined in configuration
    :param gabor_mask: which alpha mask to use, determining the shape - defined in configuration
    :param gabor_units: units to use when drawing - defined in configuration
    :param gabor_sf: spatial frequency of the Gabor - defined in configuration
    :param gabor_phase: phase of the Gabor (i.e., where are the black lines with respect to the white lines) - defined in configuration
    :param gabor_contrast: contrast of the mask - defined in configuration
    :param gabor_maskParams: sd of mask - gabor_maskSD, defined in configuration
    :param gabor_orient: which orientation Gabor should be displayed in - trial_gabor, defined in Main
    :param gabor_size: size of Gabor - defined in configuration
    :param gabor_duration: wait for pre-specified time (in frames/second) - defined in configuration

    :param mask_noiseType: which type of noise to apply (e.g., white noise) - defined in configuration
    :param mask_mask: which alpha mask to use, determining the shape - defined in configuration
    :param mask_units: units to use when drawing - defined in configuration
    :param mask_noiseElementSize: size of individual noise elements - defined in configuration
    :param mask_size: size of mask - defined in configuration
    :param mask_duration: wait for pre-specified time (in frames/second) - time_mask, defined in configuration
    :param mask_contrast: contrast for mask - defined in configuration

    :param fix_size: size of fixation cross - defined in configuration
    :param fix_lineWidth: line width of fixation cross - defined in configuration
    :param fix_ineColor: color of fixation cross - defined in configuration
    :param fix_duration: duration of fixation cross presentation - defined in configuration (time_del)
    :param fix_value: hsv color value for fixation cross - defined in configuration

    :param resp_width: width of the response bar - respBar_width, defined in configuration
    :param resp_height: height of the response bar - respBar_height, defined in configuration
    :param resp_units: units to use when drawing - respBar_units, defined in configuration
    :param respBarpool: pool of orientations from which the initial random response bar orientation is to be pulled - defined in Main
    :param respBar_color: hsv hue for response bar - defined in configuration
    :param resp_updateKeys: keys with which response bar is turned - defined in configuration/Main (keysCW_large/keysCCW_large)
    :param resp_logKeys: key with which reproduction task response is confirmed - defined in configuration (keysLogResp)
    :param keyscwccw: which keys denote 'up' and 'down' (NOT CCW/CW) in judgment task - defined in configuration (keysup/keysdown)
    
    :param iti_duration: duration of inter-trial interval - defined in Main
    
    :param eyetracking: eyetracking recorded or not?
    :param eyetracker: eyetracker object
    
    :param triggersCue: Template 1 or 2 active?
    :param triggersGabor: Gabor drawn from which orientation bin?
    :param triggersMiniblock: First, last, or middle trial in miniblock?
    :param triggersProbe: judgment task yes or no?
    :param triggersTempl1: Template 1 drawn from which orientation bin?
    :param triggersTempl2: Template 2 drawn from which orientation bin?
    :param triggersBreak: trial preceding short break?
    :param triggersPause: trial preceding long break?
    
    :param rem_templ: bool, is this the first trial of a miniblock? - defined in Main
    :param end_rememb: bool, is this the last trial of a miniblock? - defined in Main
    :param templ_width: line width of template stimuli - defined in configuration
    :param templ_height: height/radius of template stimuli - defined in configuration
    :param templ_units: units for template size - defined in configuration
    :param templ_duration: presentation duration of template stimuli - defined in configuration (time_templ)
    :param templ1_orient: orientation of template 1 - defined in Main
    :param templ2_orient: orientation of template 2 - defined in Main
    :param templ_colorspace: colorspace in which template colors are coded
    :param templ1_color: color (hsv hue) of template 1 - defined in Main
    :param templ2_color: color (hsv hue) of template 2 - defined in Main
    :param templ_value: color (hsv value) of both templates - defined in configuration

    :param cwccw_task: judgment task yes or no?
    :param trialSeq: dataframe containing trial sequence to save data during experiment (in case of abortion)
    :param triali: trial index in trial sequence of current trial
    :param feedback_1: first half of feedback text - defined in configuration (angletraining_feedback_1)
    :param feedback_2: second half of feedback text - defined in configuration (angletraining_feedback_2)
    :param last_break_triali: index in trial sequence of last trial that preceded a break - defined in this function, fed back to Main
    :param rawDat_filename: filename of dataframe containing trial sequence and responses
    :param training: training yes or no? - defined in configuration
    
    """
    from MemTempl_Taskfunctions import create_text, create_Gabor, create_Mask, create_respBar, run_stim, experiment_ratio_correct
    from MemTempl_config import text_size, text_font, text_color, angletraining_feedback_1, angletraining_feedback_2

    ##Create all the necessary stimuli
    #Create both components of both templates if it is the first trial of a miniblock
    if rem_templ == True: 
        stim_templ1circle = create_templcircle(win, templ_width, templ_height, templ_units, templ_colorspace, templ1_color, templ_value, saturation = 1)
        stim_templ2circle = create_templcircle(win, templ_width, templ_height, templ_units, templ_colorspace, templ2_color, templ_value, saturation = 1)
        stim_templ1bar = create_templbar(win, templ_width, templ_height, templ_units, templ1_orient, templ_colorspace, templ1_color, templ_value, saturation = 1) # prepare template 1 image with orientation out of trial sequence
        stim_templ2bar = create_templbar(win, templ_width, templ_height, templ_units, templ2_orient, templ_colorspace, templ2_color, templ_value, saturation = 1) # prepare template 2 image with orientation out of trial sequence

    #Create other components that are used in every trial
    stim_gabor = create_Gabor(win, gabor_tex, gabor_mask, gabor_units, gabor_sf, # prepare Gabor of correct orientation according to trial sequence
        gabor_phase, gabor_contrast, gabor_maskParams, gabor_orient, gabor_size) 
    stim_mask = create_Mask(win, mask_noiseType, mask_mask, mask_units, mask_noiseElementSize, mask_size, mask_contrast) # create mask
    stim_fix = create_fix(win, fix_size, fix_lineWidth, fix_lineColor, fix_value, templ_colorspace) # create fixation cross
    stim_cue_1 = create_cue(win, cue_size, cue_units, templ_colorspace, templ1_color, templ_value)
    stim_cue_2 = create_cue(win, cue_size, cue_units, templ_colorspace, templ2_color, templ_value)

    #Create judgment task screen and assign buttons accordingly
    if trial_ccw_up:
        keyscw = keyscwccw[0] # button 2
        keyscountcw = keyscwccw[1] # button 4
        stim_memory_probe_text = create_text(win, memory_probe_text_1, text_size, text_font, text_color)
    else:
        keyscw = keyscwccw[1] # button 4
        keyscountcw = keyscwccw[0] # button 2
        stim_memory_probe_text = create_text(win, memory_probe_text_2, text_size, text_font, text_color)
    
    cwccwkeys = [keyscw,keyscountcw]

    #Clear all keyboard events from previous trial
    kb = keyboard.Keyboard()
    win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
    win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip

    #Send any triggers/commands to eyetracker
    if eyetracking:
        if rem_templ == True:
            tracker.sendMessage('TRIALID {} {} {} {} {} {}'.format(triggersMiniblock, triggersTempl1, triggersTempl2, triggersCue, triggersGabor, triggersProbe))
        else:
            tracker.sendMessage('TRIALID {} {} {} {}'.format(triggersMiniblock, triggersCue, triggersGabor, triggersProbe))

    ##If in first trial of a miniblock, save data and run template stimuli
    if rem_templ == True: 
        #Save data up until last miniblock
        if training == 0:
            rawDat_filename_block = str(rawDat_filename) + '_final.pkl' #save as pickle
            trialSeq.to_pickle(rawDat_filename_block)
    
            rawDat_filename_block_csv = str(rawDat_filename) + '_final.csv' #save as csv
            trialSeq.to_csv(rawDat_filename_block_csv) 
            
        #Run fixation cross
        run_stim(win, stim=stim_fix, stim2=False, stim3 = False,duration=fix_duration, log_name='ITI')
        
        #Run one-word instruction ("Remember")
        run_text(win, templ_instr_text_1, text_size, text_font, text_color, duration= fix_duration, button = 'return')
        
        #Run first template stimulus, framed by fixation crosses
        if eyetracking:
            tracker.sendMessage('TRIALID {}'.format(triggersTempl1))
        
        run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
        run_stim(win, stim=stim_templ1bar,stim2=stim_templ1circle, stim3 = False,duration=templ_duration,log_name='Templ') # show Template 1
        run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
        
        #Run second template stimulus
        if eyetracking:
            tracker.sendMessage('TRIALID {}'.format(triggersTempl2))
        
        run_stim(win, stim=stim_templ2bar,stim2=stim_templ2circle, stim3 = False,duration=templ_duration,log_name='Templ') # show Template 2

        #Run two-word instruction ("Reproduce lines")
        run_text(win, templ_instr_text_2, text_size, text_font, text_color, duration= fix_duration, button= 'return')
        
        #Clear all keyboard events from previous event (in case a key is accidentally pressed during template showing)
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip

        ##Reproduce first template
        #Run fixation cross
        run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
        
        #Determine which template is reproduced first
        if templ1_cue_first_repr1:
            #Determine random initial orientation of response bar and create it
            resp_orient_templ1_1 = np.random.choice(respBarpool) # same as with template 1
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ1_1, templ_colorspace, respBar_color, templ_value, saturation = 0)
            
            #Run color cue to show subject which template to reproduce
            run_stim(win, stim=stim_cue_1, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            
            #Reproduce template orientation
            templ1_1_responses_name, templ1_1_responses_rt, templ1_1_responses_duration, templ1_1_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        else:
            #Determine random initial orientation of response bar and create it
            resp_orient_templ2_1 = np.random.choice(respBarpool) # same as with template 1
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ2_1, templ_colorspace, respBar_color, templ_value, saturation = 0)
            
            #Run color cue to show subject which template to reproduce
            run_stim(win, stim=stim_cue_2, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            
            #Reproduce template orientation
            templ2_1_responses_name, templ2_1_responses_rt, templ2_1_responses_duration, templ2_1_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        
        #Clear all keyboard events from previous event
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
        
        ##Reproduce second template (same procedure)
        run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
        
        if templ1_cue_first_repr1:
            resp_orient_templ2_1 = np.random.choice(respBarpool) # same as with template 1
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ2_1, templ_colorspace, respBar_color, templ_value, saturation = 0)
            run_stim(win, stim=stim_cue_2, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            templ2_1_responses_name, templ2_1_responses_rt, templ2_1_responses_duration, templ2_1_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        else:
            resp_orient_templ1_1 = np.random.choice(respBarpool) # same as with template 1
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ1_1, templ_colorspace, respBar_color, templ_value, saturation = 0)
            run_stim(win, stim=stim_cue_1, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            templ1_1_responses_name, templ1_1_responses_rt, templ1_1_responses_duration, templ1_1_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        
        #Clear all keyboard events from previous event
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
        
        #Run two-word instruction ("Reproduce patches")
        run_text(win, inst_reproduction, text_size, text_font, text_color, duration=fix_duration, button='return')
            
    #In all in-between trials, produce NaN to insert into template-related response dataframe cells
    else:
        templ1_1_responses_name = np.nan
        templ1_1_responses_rt = np.nan
        templ1_1_responses_duration = np.nan
        templ1_1_angle = np.nan
        templ2_1_responses_name = np.nan
        templ2_1_responses_rt = np.nan
        templ2_1_responses_duration = np.nan
        templ2_1_angle = np.nan
        resp_orient_templ1_1 = np.nan
        resp_orient_templ2_1 = np.nan
    
    #Run fixation cross
    run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
    
    #Run color cue
    if trial_active == 1:
        run_stim(win, stim=stim_cue_1, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
    elif trial_active == 2:
        run_stim(win, stim=stim_cue_2, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
    
    if eyetracking:
        tracker.sendMessage('TRIALID {}'.format(triggersGabor))
    
    #Run fixation cross, Gabor stimulus, mask stimulus, and second fixation cross
    run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
    run_stim(win, stim=stim_gabor, stim2=False, stim3=False, duration=gabor_duration, log_name='Gabor')
    run_stim(win, stim=stim_mask, stim2=False, stim3=False, duration=mask_duration, log_name='Mask')
    run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=fix_duration, log_name='Delay')
    
    #Determine random initial orientation of response bar
    resp_orient_item = np.random.choice(respBarpool)
    stim_respBar = create_respBar(win, bg_color, resp_width, resp_height, resp_units, resp_orient_item)
    
    #Reproduce Gabor orientation
    mem_item_responses_name, mem_item_responses_rt, mem_item_responses_duration, mem_item_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)

    #Clear all keyboard events from previous event
    kb = keyboard.Keyboard()
    win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
    win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
       
    #Run judgment task screen and record response
    if cwccw_task == True:
        cwccw_resp, cwccw_rt = cwccw_stim(win, stim=stim_memory_probe_text, stim2=False, keyscwccw = cwccwkeys) # response screen
    else:
        cwccw_resp = np.nan
        cwccw_rt = np.nan
     
    ##If in last trial of a miniblock, run template stimuli
    if end_rememb == True:
        run_text(win, post_templ_text, text_size, text_font, text_color, duration=fix_duration, button='return')
        
        #Reproduce first template (same procedure as above)
        if templ1_cue_first_repr2: 
            resp_orient_templ1_2 = np.random.choice(respBarpool)
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ1_2, templ_colorspace, respBar_color, templ_value, saturation = 0)
            run_stim(win, stim=stim_cue_1, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            templ1_2_responses_name, templ1_2_responses_rt, templ1_2_responses_duration, templ1_2_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        else: 
            resp_orient_templ2_2 = np.random.choice(respBarpool)
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ2_2, templ_colorspace, respBar_color, templ_value, saturation = 0)
            run_stim(win, stim=stim_cue_2, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            templ2_2_responses_name, templ2_2_responses_rt, templ2_2_responses_duration, templ2_2_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
       
        #Clear all keyboard events from previous event
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
        
        #Reproduce second template (same procedure as above)
        if templ1_cue_first_repr2:
            run_stim(win, stim=stim_cue_2, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            resp_orient_templ2_2 = np.random.choice(respBarpool)
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ2_2, templ_colorspace, respBar_color, templ_value, saturation = 0)
            templ2_2_responses_name, templ2_2_responses_rt, templ2_2_responses_duration, templ2_2_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
       
        else:
            run_stim(win, stim=stim_cue_1, stim2=False, stim3=False, duration=fix_duration, log_name='Cue')
            resp_orient_templ1_2 = np.random.choice(respBarpool)
            stim_respBar = create_templbar(win, templ_width, templ_height, templ_units, resp_orient_templ1_2, templ_colorspace, respBar_color, templ_value, saturation = 0)
            templ1_2_responses_name, templ1_2_responses_rt, templ1_2_responses_duration, templ1_2_angle = update_stim(win, stim=stim_respBar, stim2=False, updateKeys=resp_updateKeys, logKeys=resp_logKeys)
        
        #Clear all keyboard events from previous event
        kb = keyboard.Keyboard()
        win.callOnFlip(kb.clock.reset)  #t=0 on next screen flip
        win.callOnFlip(kb.clearEvents, eventType='keyboard')  #clear events on next screen flip
        
        #Run fixation cross
        run_stim(win, stim=stim_fix, stim2=False, stim3=False, duration=iti_duration, log_name='ITI')
    
    #In all in-between trials, produce NaN to insert into template-related response dataframe cells
    else:
        templ1_2_responses_name = np.nan
        templ1_2_responses_rt = np.nan
        templ1_2_responses_duration = np.nan
        templ1_2_angle = np.nan
        templ2_2_responses_name = np.nan
        templ2_2_responses_rt = np.nan
        templ2_2_responses_duration = np.nan
        templ2_2_angle = np.nan
        resp_orient_templ1_2 = np.nan
        resp_orient_templ2_2 = np.nan
        
    ##End trial in feedback followed by break screen in case of break trials
    if break_trial == True:
        if eyetracking:
            tracker.sendMessage('TRIALID {}'.format(triggersBreak))
        
        #Determine ratio of correct judgment task responses since last break (or since beginning in case of first break)
        ratio = experiment_ratio_correct(trialSeq[last_break_triali:triali])
        
        #Generate feedback text with individual ratio
        feedback_text = angletraining_feedback_1 + str(ratio) + angletraining_feedback_2

        #Run feedback and break screens (to move between at subject's pace)
        run_text(win, feedback_text, text_size, text_font, text_color, duration='inf', button='return')
        run_text(win, break_text_short, text_size, text_font, text_color, duration='inf', button='return')
        
        #Record latest break trial
        last_break_triali = triali
    
    ##End trial in feedback followed by break screen in case of pause trials
    #(same procedure as with break trials, only difference is text of break screen)
    if pause_trial == True:
        if eyetracking:
            tracker.sendMessage('TRIALID {}'.format(triggersPause))
        
        ratio = experiment_ratio_correct(trialSeq[last_break_triali:triali])

        feedback_text = angletraining_feedback_1 + str(ratio) + angletraining_feedback_2
        
        run_text(win, feedback_text, text_size, text_font, text_color, duration='inf', button='return')
        run_text(win, break_text_long, text_size, text_font, text_color, duration='inf', button='return')
        
        last_break_triali = triali
        
    ##End trial in fixation cross of duration specified in dataframe (ITI)
    else:
        run_stim(win, stim=stim_fix, stim2=False, stim3 = False,duration=iti_duration, log_name='ITI')

    ##Return all responses and parameters given or generated in the trial
    return last_break_triali,templ1_1_responses_name, templ1_1_responses_rt, templ1_1_responses_duration, templ1_1_angle, templ2_1_responses_name, templ2_1_responses_rt, templ2_1_responses_duration, templ2_1_angle, mem_item_responses_name, mem_item_responses_rt, mem_item_responses_duration, mem_item_angle, templ1_2_responses_name, templ1_2_responses_rt, templ1_2_responses_duration, templ1_2_angle, templ2_2_responses_name, templ2_2_responses_rt, templ2_2_responses_duration, templ2_2_angle, cwccw_resp, cwccw_rt, resp_orient_templ1_1, resp_orient_templ2_1,resp_orient_templ1_2, resp_orient_templ2_2, resp_orient_item