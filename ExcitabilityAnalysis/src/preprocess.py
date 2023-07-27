
#%%
# Import libraries
import os
from helper_preprocess import *

rootdir = "C:\\Users\\Lab\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability"
subject_folders = os.listdir(rootdir)
subject_keypaths = [[]] * len(subject_folders)

for i in range(n_subjects):
    #subject_keypaths[i] = 'test'
    subject_dir = os.path.join(rootdir, subject_folders[i])
    print(subject_dir)
    for subject_dir, dirs, files in os.walk(subject_dir):
        for subdir in dirs:
            print(subdir)
            subject_keypaths[i].append(os.path.join(subject_dir, subdir))
            print(os.path.join(subject_dir, subdir))
            
#work on adding keypaths to different sections of the list


SID_list = []
key_paths = []
for rootdir, dirs, files in os.walk(rootdir):
    #print("directory", dirs)
    for subdir in dirs:
        #print("subdirectory", subdir)
        regex = "[0-9]{8}$"
        ID_datecheck = re.findall(regex, subdir)
        if ID_datecheck:
            key_paths.append(os.path.join(rootdir, subdir))
            #print(key_paths)
        else: 
            SID_list.append(subdir)

#print(key_paths)
#print(SID_list)
#df_ActSCS, idxs_ActSCS, df_ActRest, idxs_ActRest, df_SCS, idxs_SCS  = trainingDFs(key_paths)


    