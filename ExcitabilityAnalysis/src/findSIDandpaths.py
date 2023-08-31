#%%
# Import libraries
import os
import re
import numpy as np
import pandas as pd
from helper_preprocess import trainingDFs, readcsvs_concatdf

rootdir = "C:\\Users\\Lab\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability"
#rootdir = "C:\\Users\\marie\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability"

subject_folders = os.listdir(rootdir)
dict_SID_path = {}

for i in range(len(subject_folders)):
    subject_dir = os.path.join(rootdir, subject_folders[i])
    name = ''
    holder = []
    for subject_dir_date, dirs, files in os.walk(subject_dir):
        for subdir in dirs:
            name = os.path.join(subject_dir_date, subdir)
            holder.append(name)
    dict_SID_path[subject_folders[i]] = holder


df_ActSCS, idxs_ActSCS, df_ActRest, idxs_ActRest, df_SCS, idxs_SCS  = trainingDFs(dict_SID_path['EX002'])
print("Walking through keypaths.")

# %%
