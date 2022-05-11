#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Purpose: This file contains basic functions needed to run the eyelink eyetrackr
#Author: Darinka Truebutschek
#Date created: 25/08/2021
#Date last modified: 25/08/2021
#Python version: 3.7.1

#Göttingen: Eyelink1000+
#pylink 0.3.3 does not seem to work

def setupEyetracker(subject, session, win, mon_size):
    '''
    :param subject: name of the subject
    :param session: experimental session for particular subject
    :param win: window to be used
    :param mon_size: monitor size (in pixels)
    '''

    #Import library
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

    #Connect to the eyetracker
    tk = pylink.EyeLink('100.1.1.1')

    #Create an EDF file to save eyetracking data to (cannot exceed 8 charaters)
    filename_edf = subject + session + ".edf" #subject + session + '.EDF'
    tk.openDataFile(filename_edf)
    tk.sendCommand("add_file_preamble_text = 'Eyetracker data - SerDep_EvBound'")

    #Initialize the graphics for calibration/validation
    g_env = EyeLinkCoreGraphicsPsychoPy(tk, win)
    pylink.openGraphicsEx(g_env)

    #Put the eyetracker into offline mode before changing its configuration
    tk.setOfflineMode()

    #Set sampling rate (this command won't work for EyeLink II/I)
    tk.sendCommand("sample_rate 1000")

    #Inform the eyetracker about the resolution of the monitor [see EyeLink Installation Guide, Section 8.4: Customizing your PHYSICAL.INI Settings]
    tk.sendCommand("screen_pixel_coords = 0 0 %d %d" % (mon_size[0]-1, mon_size[1]-1))
    
    #Setting eyetracker to binocular 
    tk.sendCommand("binocular_enabled = NO")

    #Save display resolution in EDF data file for Data Viewer integration purposes [see Data Viewer User Manual, Section 7: Protocol for EyeLink Data to Viewer Integration]
    tk.sendMessage("DISPLAY_COORDS = 0 0 %d %d" % (mon_size[0]-1, mon_size[1]-1))

    #Specify the calibration type: H3, HV3, HV5, HV13 (HV=horizontal/vertical)
    #tk.sendCommand("calibration_type = HV13") #13-point calibration

    #Get the model of the eyetracker: 1-EyeLink I, 2-EyeLink II, 3-Newer models(100/1000Plus/DUO)
    eyelinkVer = tk.getTrackerVersion()

    #Turn off scenelink camera for EyeLink II/I only
    if eyelinkVer == 2:
        tk.sendCommand("scene_camera_gazemap = NO")

    #Set up the tracker to parse events using "GAZE" (or "HREF") data
    tk.sendCommand("recording_parse_type = GAZE")

    #Online parseer configuration: 0 -> standard/cognitive, 1 -> sensitive/psychophysiologial
    if eyelinkVer >= 2:
        tk.sendCommand("select_parser_configuration 0")

    #Get host tracking software version
    hostVer = 0
    if eyelinkVer == 3:
        tvstr = tk.getTrackerVersionString()
        vindex = tvstr.find("EYELINK CL")
        hostVer = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))

    #Specify the event and sample data that are stored in EDF 
    tk.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
    tk.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
    if hostVer >= 4:
        tk.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
        tk.sendCommand("link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
    else:
        tk.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")
        tk.sendCommand("link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")

    #Specify calibration type (number of points)
    #tk.sendCommand("calibration_type = HV13")
    tk.sendCommand("calibration_type = HV9") #maybe this doesn't work
    #tk.sendCommand("calibration_type = HV5") #this was tested; works
    
    #Turn off sounds
    #pylink.setCalibrationSounds("off", "off", "off")
    #pylink.setDriftCorrectSounds("off", "off", "off")
    
    #Reduce area in which calibration points are displayed
    tk.sendCommand("calibration_area_proportion = 0.40 0.40") #has worked before
    #tk.sendCommand("calibration_targets = 960,540 960,376 960,704 743,540 1180,540") #hasn't been tested, is maybe needed for previous line to be effective?
    
    #tk.sendCommand("draw_box %d %d %d %d 15", FixSquare(1), FixSquare(2), FixSquare(3), FixSquare(4))
    #Eyelink('command', 'draw_box %d %d %d %d 15',  FixSquare(1), FixSquare(2),FixSquare(3), FixSquare(4))

    return tk, filename_edf

def calibrateEyetracker(tk, win):
    '''
    :param tk: eyetracker object
    :param win: window to be used
    '''

    from psychopy import event, visual

    continueLoop = True
    while continueLoop:
        instructions_cal = visual.TextStim(win, text='Press <ENTER> twice to calibrate the eyetracker', color='White')
        instructions_cal.draw()
        win.flip()
        event.waitKeys()

        tk.doTrackerSetup()
        eyetracker_text = visual.TextStim(win, text='Press <E> to run the eyetracker again, or press <C> to run the experiment.', height=0.5, units='deg', color='White')
        
        win.flip()
        eyetracker_text.draw()
        win.flip()

        p_keys = event.waitKeys(keyList=['e', 'c'], clearEvents=True)

        if p_keys[0] == 'e':
            continueLoop = True
            win.flip()
        elif p_keys[0] == 'c':
            continueLoop = False
            win.flip()

    return tk

def quitEyetracker(tk, edf_filename, win, path_edfDat):
    '''
    :param tk: eyetracker object
    :param edf_filename: savename for eyetracking data
    :param win: window to be used
    :param edfDat: where to store the eyetracking data
    '''

    from psychopy import visual
    from pathlib import Path

    #Close EDF file
    tk.closeDataFile()

    #EyeLink - copy EDF file to display pc and save it in local folder ('edfData')
    edfTransfer = visual.TextStim(win, text='Transfering eyetracking data ...', color='White')
    edfTransfer.draw()

    win.flip()

    tk.receiveDataFile(edf_filename, str(path_edfDat / edf_filename))

    #Close connection to the eyetracker
    tk.close()


