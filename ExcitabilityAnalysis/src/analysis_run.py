#Code to run full analysis pipeline
from preprocess_csv import *
import pandas as pd
import os
import sys
from pathlib import Path

#Run preprocess_csv to get Fwave and TMS data
userPath = os.path.expanduser('~') #gets path to user on any computer
boxPath = Path(userPath + 'Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability') #puts user into box

root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"
savepath = "C:/Users/Lab/Documents/ExcitabilityAnalysis/data/raw"

fwave_list, tms_list = preprocess_main(root)

#Run fwave_analysis.py to return all Fwave metrics and plots
#fwave_output = fwave_metrics(fwave_list, savepath)
    #fwave_output.plot
    #fwave_output.stats

#Run tms_analysis.py to return all MEP metrics and plots
#tms_output = tms_metrics(tms_list, savepath)
    #tms_output.plot
    #tms_output.stats