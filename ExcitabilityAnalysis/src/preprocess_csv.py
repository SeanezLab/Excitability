#%% Read in all csv files, add dataframes, sort between fwave and tms, add in column names to all dataframes
from pathlib import Path
import numpy as np
import pandas as pd
import re
import glob
import os

def get_SID_subdir(path):
    """https://perials.com/getting-csv-files-directory-subdirectories-using-python/"""
    root = path
    dir_list = []
    split_list = []
    SID = []
    csv_list = []
    
    for base, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            regex = "[0-9]{8}$"
            ID_date_check = re.findall(regex, name)
            if ID_date_check:
                dir_name = os.path.join(root, name)
                base_name = os.path.basename(dir_name)
                split_name = base_name.split('_')
                SID.append(split_name[0])
                split_list.append(split_name)
                dir_list.append(os.path.join(root, name))
    
    home_dir = str(os.path.dirname(dir_list[0]))
    os.chdir(home_dir)
    csv_files = glob.glob('**/*.csv', recursive=True)
    dir_csv = []
    for i in range(len(csv_files)): #add home directory to .csv path
        dir_csv.append(os.path.abspath(csv_files[i]))
    
    df_ID_date = pd.DataFrame(split_list, columns=['SID', 'DATE'])
    df_ID_date['CSV'] = 0
 
    for j in range(len(SID)):
        SID_value = SID[j]
        temp_csv_list = []
        for k in range(len(dir_csv)):
            file_name = dir_csv[k]
            SID_check = re.search(SID_value, file_name) 
            if SID_check:
                temp_csv_list.append(file_name)
      
        df_ID_date['CSV'] = df_ID_date['CSV'].astype('object')
        df_ID_date.at[j, 'CSV'] = temp_csv_list
        
        csv_list.append(temp_csv_list)

    return dir_list, df_ID_date, csv_list


def read_csv(files): 
    """https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/"""
    # create empty list
    dataframes_list = []
    # append datasets into the list
    for i in range(len(files)):
        basename = os.path.splitext(os.path.basename(files[i]))[0]
        print(basename)
        
        if basename == 'sensors':
            temp_df = pd.read_csv(files[i], header=None)
            temp_df.reset_index(drop=True)

        else:
            temp_df = pd.read_csv(files[i])
            dfname = basename.replace('Configuration_', '').replace('_1', '')
            temp_df.index.name = dfname
            temp_df.reset_index(drop=True)
            
        dataframes_list.append(temp_df)
    return dataframes_list

   
def rename_df_columns(dataframes_list):
    EMG_sensorlist = list(dataframes_list[-1][2])
    print(EMG_sensorlist)
    
    for i in range(len(dataframes_list)):
        df = dataframes_list[i]
        ID = df.index.name
        print(ID)
        if ID != "sensors":
            #rename headers in df and save
            existing_column_names = list(df.columns.values)
            sensor_names = EMG_sensorlist #get sensor names from sensor list that correspond to muscles
            replace_names = dict(zip(existing_column_names, sensor_names)) #create dictionary mapping old names to new names
            df_renamed = df.rename(replace_names, axis='columns') #rename df columns
            df_renamed.reset_index(inplace=True)
            df_renamed.index.name = ID
            
    return df_renamed

# def rename_all_columns(sensors_lists, df_fwave, df_tms):
#     """"Add all sensor names to column names across all dataframes
#     Could become obselate if csv is saved in a different fashion to 
#     automatically add sensor names as the column names"""
    
#     fwave_renamed = []
#     tms_renamed = []
#     print(sensors_lists)
#     for i in range(len(sensors_lists)):
#         regex = "Fwave"
#         name = re.findall(regex, sensors_lists[i])
#         if name:
#             for j in range(len(df_fwave)):
#                 temp_fwave = rename_df_columns(sensors_lists[i], df_fwave[j])
#                 fwave_renamed.append(temp_fwave)
#         else:
#             for k in range(len(df_tms)):
#                 temp_tms = rename_df_columns(sensors_lists[i], df_tms[k])
#                 tms_renamed.append(temp_tms)
#     return fwave_renamed, tms_renamed


# def df_sort(df_list):
#     """Sort between Fwave dataframes and TMS dataframes for later analysis"""
#     df_fwave = []
#     df_tms = []
    
#     for i in range(len(df_list)):
#         regex = "Fwave"
#         name = re.findall(regex, df_list[i].index.name)
#         if name:
#             df_fwave.append(df_list[i])
#         else:
#             df_tms.append(df_list[i])
#     return df_fwave, df_tms

# def preprocess_main(root):
#     root = root
#     #read all .csv files of interest, and sort for csv with EMG data
#     EXT = '*.csv'
#     regex = "EX0[0-9][0-9]_"
#     csv_files, sorted_csvs = get_csv(root, EXT, regex)
#     #read csv into dataframe
#     df_list = read_csv(sorted_csvs)
#     sensors_lists = get_sensors(csv_files)
#     df_fwave, df_tms = df_sort(df_list)
#     df_fwave_vf, df_tms_vf = rename_all_columns(sensors_lists, df_fwave, df_tms)

#     print("All done ✨")
    
#     return df_fwave_vf, df_tms_vf
    
root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"
dir_list, df_ID_date, all_csv_list = get_SID_subdir(path = root)
SID = df_ID_date['SID']

temp = read_csv(all_csv_list[0])
named = rename_df_columns(temp)


# all_csv_df = []
# for i in range(len(SID)):
#     SID_name = SID[i]    
#     temp = read_csv(all_csv_list[i])
#     print(temp.index.name)
    #all_csv_df.append([SID_name, temp])
    



