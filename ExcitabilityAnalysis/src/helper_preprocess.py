"""
Created on Thu Jul 27 14:41:25 2023

@author: Rachel Hawthorn
"""
import glob
import pandas as pd
import os

def readcsvs_concatdf(csv_files):
    """
    Send in list of absolute paths of csv files 
    Function will 
    - create large concatenated df of all csv data
    - add muscle names to all corresponding columns
    - add file basenames to index for later sectioning
    Return large df, and list of string indices of basenames
    """
    
    df_list = []
    csv_index = []

    for file in csv_files:
        if "sensor" in file:
            sensor_info = pd.read_csv(file, header=None)
            muscle_names = list(sensor_info[2])
            #print("Muscle Names", muscle_names)

    for file in csv_files:
        if "sensor" in file:
            print(" ")
        else:   
            temp = pd.read_csv(file, names=muscle_names, skiprows=1)
            basename = os.path.splitext(os.path.basename(file))[0]
            temp.index.name = basename #rename index of temp to key info from filename
            csv_index.append(basename)
            df_list.append(temp)

    df = pd.concat(df_list, keys=csv_index)
    
    return df, csv_index
            
def trainingDFs(key_paths):
    """
    Send in list of absolute paths of folders that contain csvs
    Function will 
    - call readcsvs_concatdf
    Return large df of each training day, and list of string indices of basenames
    """
    for path in key_paths:
        path = r'{}'.format(path)
        csv_files = glob.glob(path + "\*.csv")
        df_check = os.path.splitext(os.path.basename(csv_files[0]))[0]
     
        if "_ActSCS" in df_check:
            print("BoMi and SCS Training Day")
            df_ActSCS, idxs_ActSCS = readcsvs_concatdf(csv_files)
            print("Done")
        elif "_ActRest" in df_check:
            print("BoMi and Rest Training Day")
            df_ActRest, idxs_ActRest = readcsvs_concatdf(csv_files)
            print("Done")

        elif "_SCS" in df_check:
            print("SCS Training Day")
            df_SCS, idxs_SCS = readcsvs_concatdf(csv_files)
            print("Done")

        else:
            print("No matching test conditions")
        
    print("All done ✨")
    return df_ActSCS, idxs_ActSCS, df_ActRest, idxs_ActRest, df_SCS, idxs_SCS 
