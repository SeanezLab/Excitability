#%% 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import re
from glob import glob
import os

def get_csv(path, ext, regex):
    """https://perials.com/getting-csv-files-directory-subdirectories-using-python/"""
    all_csv_files = []
    for path, subdir, files in os.walk(root):
        for file in glob(os.path.join(path, EXT)):
            all_csv_files.append(file)
            #print(all_csv_files)
    sorted_files = []
    """ """
    for i in range(len(all_csv_files)):
        #pull csv that have "EX00#_ in the name --> EMG data files
        name = re.findall(regex, os.path.basename(all_csv_files[i]))
        if name:
            sorted_files.append(all_csv_files[i])
        print(sorted_files)
    return all_csv_files, sorted_files

def read_csv(files): 
    """https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/"""
    # create empty list
    dataframes_list = []
    # append datasets into the list
    for i in range(len(files)):
        index_name = files[i].split("\\")[8] #pulls EX001_Fwave or _TMS to set as index
        print(index_name)
        temp_df = pd.read_csv(files[i])
        temp_df.index.name = index_name
        dataframes_list.append(temp_df)
    return dataframes_list

#%%
if __name__ == "__main__":

    root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"

    #read all .csv files of interest, and sort for csv with EMG data
    EXT = '*.csv'
    #regex = "EX0[0-9][0-9]_"
    regex = "EX001_PNS"
    csv_files, sorted_csvs = get_csv(root, EXT, regex)
    #print(sorted_csvs)
    #read csv into dataframe
    df_list = read_csv(sorted_csvs)

#%%
EX001_PNS = df_list[0]

sensor_filepath = 'C:\\Users\\Lab\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability\\EX001_PNS\\EX001_PNS_20230601\\sensors.csv'
EMG_sensorlist = pd.read_csv(sensor_filepath, header=None)

#rename headers in EX003_TMS dataframe and save to TMS df
existing_column_names = list(EX001_PNS.columns.values)
sensor_names = list(EMG_sensorlist[2]) #get sensor names from sensor list that correspond to muscles
replace_names = dict(zip(existing_column_names, sensor_names)) #create dictionary mapping old names to new names
df_Fwave = EX001_PNS.rename(replace_names, axis='columns') #rename df columns
df_Fwave.reset_index(inplace=True)

df_Fwave
#%%
from scipy.signal import find_peaks
import matplotlib as mpl
#%matplotlib widget

trigger_channel = np.array(df_Fwave['Trigger'])

right_soleus = np.array(df_Fwave['R Soleus'])

cut_trigger = (trigger_channel[2900000:])*-1
cut_soleus = right_soleus[2900000:] * (10**3) 
cut_soleus = cut_soleus - np.mean(cut_soleus)     

#cut_trigger = cut_trigger[132000:220000]
#cut_soleus = cut_soleus[132000:220000]


height = np.mean(cut_trigger) + (3*np.std(cut_trigger[0:20000])) #height requirement for peak detection
peaks, _ = find_peaks(cut_trigger, height=height, distance=500)

#plt.plot(cut_trigger) #plot trigger signal
#plt.plot(peaks, cut_trigger[peaks], "x", markersize = 10) #plot markers on peaks
#plt.axhline(y=height, linestyle = '--', color = 'k')
 
starting_idx = peaks + 130
ending_idx = peaks + 220

sampling_rate = 4400 #Hz, or samples/sec
sr_roi_samples = int(sampling_rate * 0.24) #number of samples for 0.24 seconds
trigger_idx = int(0.04 * sampling_rate)
time = sr_roi_samples/sampling_rate
total_time = 90/4400 #15.9091 ms
x = np.linspace(0, total_time*1000, 90)  



#%%
# import required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

fs = 4400  # Sampling frequency
fc = 200  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency

trials = []


plt.figure(1)
for i in range(len(peaks)):
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, (cut_soleus[starting_idx[i]:ending_idx[i]]))
    plt.plot(x, output, color ='k', alpha=0.6, linewidth = 0.2)
    plt.ylabel('Millivolts')
    plt.xlabel('Milliseconds')
    plt.title('LPF @ 200 Hz')
    plt.grid(visible = True)
    trials.append(cut_soleus[starting_idx[i]:ending_idx[i]])

h = np.array(trials)
b, a = signal.butter(5, w, 'low')
res = np.average(h, axis=0)
res_std = np.std(h, axis=0)

res_low = signal.filtfilt(b, a, res)

plt.plot(x, res_low, color = 'green')
plt.show()


#%%
plt.figure(2)
for i in range(len(peaks)):
    plt.plot(x,(cut_soleus[starting_idx[i]:ending_idx[i]])-res_low, color ='k', alpha=0.8, linewidth = 0.2)
    plt.plot(x, res, color = 'green')
    plt.ylim(-0.03, 0.03)
    plt.pause(0.1)
    
    if i != len(peaks) - 1:
        plt.waitforbuttonpress()
        plt.clf()
    else:
       plt.show()

plt.ylabel('Millivolts')
plt.xlabel('Milliseconds')
plt.title('Raw Traces - LPF Signal Avg')
plt.close()

#%%
#plt.figure(3)
offset = np.linspace(0,0.8,20)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks(offset)

sns.set()
sns.set_style("white")
custom_palette = sns.dark_palette("seagreen", 20)

for i in range(len(peaks)):
    plt.plot(x,(cut_soleus[starting_idx[i]:ending_idx[i]])+offset[i]-res,linewidth = 2,
             color = custom_palette[i],alpha=0.8)
    plt.axhline(offset[i], color='k',alpha=0.15)

plt.title('Raw F-wave Traces - Average (Soleus)')
plt.ylabel('mV')
plt.xlabel('ms')
plt.show()

#%%
#plt.figure(3)
offset = np.linspace(0,.7,20)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks(offset)

sns.set()
sns.set_style("white")
custom_palette = sns.dark_palette("seagreen", 20)

for i in range(len(peaks)):
    plt.plot(x,(cut_soleus[starting_idx[i]:ending_idx[i]])+offset[i]-res_low,linewidth = 2,
             color = custom_palette[i],alpha=0.8)
    plt.axhline(offset[i], color='k',alpha=0.15)

plt.title('Raw F-wave Traces - LowPass Average Trace')
plt.ylabel('mV')
plt.xlabel('ms')
plt.show()

#%%

plt.figure(6)
offset = np.linspace(0,1.5,20)

starting_idx = peaks - 50
ending_idx = peaks + 220
total_time = 270/4400 #15.9091 ms
x = np.linspace(0, total_time*1000, 270)  

for i in range(len(peaks)):
    plt.plot(x,(cut_soleus[starting_idx[i]:ending_idx[i]]) +offset[i], color ='k', linewidth = 0.2)


plt.ylabel('mV')
plt.xlabel('ms')
plt.show()








#%%
#TMS pre and post
EX001_TMS = df_list[1]
EX001_TMS_2 = df_list[2]

sensor_filepath = 'C:\\Users\\Lab\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability\\EX001_TMS\\EX001_TMS_20230601\\sensors.csv'
EMG_sensorlist = pd.read_csv(sensor_filepath, header=None)

#rename headers in EX003_TMS dataframe and save to TMS df
existing_column_names = list(EX001_TMS.columns.values)
sensor_names = list(EMG_sensorlist[2]) #get sensor names from sensor list that correspond to muscles
replace_names = dict(zip(existing_column_names, sensor_names)) #create dictionary mapping old names to new names
df_TMS = EX001_TMS.rename(replace_names, axis='columns') #rename df columns
df_TMS.reset_index(inplace=True)


sensor_filepath = 'C:\\Users\\Lab\\Box\\Seanez_Lab\\SharedFolders\\RAW DATA\\Excitability\\EX001_TMS_2\\EX001_TMS_2_20230601\\sensors.csv'
EMG_sensorlist = pd.read_csv(sensor_filepath, header=None)

#rename headers in EX003_TMS dataframe and save to TMS df
existing_column_names = list(EX001_TMS_2.columns.values)
sensor_names = list(EMG_sensorlist[2]) #get sensor names from sensor list that correspond to muscles
replace_names = dict(zip(existing_column_names, sensor_names)) #create dictionary mapping old names to new names
df_TMS_2 = EX001_TMS_2.rename(replace_names, axis='columns') #rename df columns
df_TMS_2.reset_index(inplace=True)


#%%
#PRE TMS EX001

trigger_channel_tms = np.array(df_TMS['Trigger'])
rSOL = np.array(df_TMS['R Soleus'])

cut_trigger = (trigger_channel_tms[3700000:])*-1
cut_soleus = rSOL[3700000:] * (10**6) #put in uV
cut_soleus = cut_soleus - np.mean(cut_soleus)

height = np.mean(trigger_channel_tms[1000000:1300000]) + (3*np.std(trigger_channel_tms[1000000:1300000])) #height requirement for peak detection
peaks, _ = find_peaks(cut_trigger, height=height, distance=500)


starting_idx = peaks - 10
ending_idx = peaks + 300

total_time = 310/4400 #4400 #Hz, or samples/sec
x = np.linspace(0, total_time*1000, 310)  

trials = []

plt.figure(4)
for i in range(len(peaks)):
    plt.plot(x, cut_soleus[starting_idx[i]:ending_idx[i]], color ='k', alpha=0.4, linewidth = 0.2)
    plt.ylabel('MicroVolts (uV)')
    plt.xlabel('Milliseconds (ms)')
    plt.title('Right Soleus Pre, 1.2xRMT: 61%')
    #plt.grid(visible = True)
    trials.append(cut_soleus[starting_idx[i]:ending_idx[i]])

h = np.array(trials)
res = np.average(h, axis=0)
res_std = np.std(h, axis=0)

peaks2, _= find_peaks(res, height = 10, distance = 200)
troughs, _= find_peaks(-res,height = 10,  distance = 200)

plt.axvline(x = x[10], linestyle = '--', color = 'gray', linewidth = 0.8, label = "Stim Pulse")
plt.plot(x, res, color = 'green', label = "Avg Trace")
p_idx = x[peaks2[0]]
y_idx = res[peaks2[0]]

t_idx = x[troughs[0]]
ty_idx = res[troughs[0]]
print('_______________________________________')
print('PRE')
print('Peak Max: ' + str(round(y_idx, 2)) + 'uV', 'and Latency: ' + str(round(p_idx, 2)) + 'ms')
print('Peak Min: ' + str(round(ty_idx, 2)) + 'uV', 'and Latency: ' + str(round(t_idx, 2)) + 'ms')

plt.scatter(p_idx, y_idx, s=50, marker = '^', color = 'k', label = 'Peak Max: ' + str(round(y_idx, 2)) + ' uV')
plt.scatter(t_idx, ty_idx, s=50, marker = 'v', color = "k", label = 'Peak Min: ' + str(round(ty_idx, 2)) + ' uV')


plt.fill_between(x, (res-res_std), (res+res_std), facecolor = 'green', alpha = 0.18)

amp = y_idx-ty_idx
print('Peak to Peak Amplitude: ' + str(round(amp,2)) + 'uV')
plt.legend(title = ('Amplitude: ' + str(round(amp,2)) + 'uV'))

plt.show()


#%%
#POST TMS EX001

trigger_channel_tms = np.array(df_TMS_2['Trigger'])
rSOL = np.array(df_TMS_2['R Soleus'])


cut_trigger = (trigger_channel_tms[1800000:])*-1
cut_soleus = rSOL[1800000:] * (10**6) #put in uV
cut_soleus = cut_soleus - np.mean(cut_soleus)

height = np.mean(trigger_channel_tms[1700000:1800000]) + (3*np.std(trigger_channel_tms[1700000:1800000])) #height requirement for peak detection
peaks, _ = find_peaks(cut_trigger, height=height, distance=500)


starting_idx = peaks - 10
ending_idx = peaks + 300

total_time = 310/4400 #4400 #Hz, or samples/sec
x = np.linspace(0, total_time*1000, 310)  

trials = []

plt.figure(5)
for i in range(len(peaks)):
    plt.plot(x, cut_soleus[starting_idx[i]:ending_idx[i]], color ='k', alpha=0.4, linewidth = 0.2)
    plt.ylabel('MicroVolts (uV)')
    plt.xlabel('Milliseconds (ms)')
    plt.title('Right Soleus Post, 1.2xRMT: 61%')
    #plt.grid(visible = True)
    trials.append(cut_soleus[starting_idx[i]:ending_idx[i]])

h = np.array(trials)
res = np.average(h, axis=0)
res_std = np.std(h, axis=0)

peaks2, _= find_peaks(res, height = 10, distance = 200)
troughs, _= find_peaks(-res,height = 10,  distance = 200)

plt.axvline(x = x[10], linestyle = '--', color = 'gray', linewidth = 0.8, label = "Stim Pulse")
plt.plot(x, res, color = 'green', label = "Avg Trace")
p_idx = x[peaks2[0]]
y_idx = res[peaks2[0]]

t_idx = x[troughs[0]]
ty_idx = res[troughs[0]]
print('_______________________________________')
print('POST')
print('Peak Max: ' + str(round(y_idx, 2)) + 'uV', 'and Latency: ' + str(round(p_idx, 2)) + 'ms')
print('Peak Min: ' + str(round(ty_idx, 2)) + 'uV', 'and Latency: ' + str(round(t_idx, 2)) + 'ms')

plt.scatter(p_idx, y_idx, s=50, marker = '^', color = 'k', label = 'Peak Max: ' + str(round(y_idx, 2)) + ' uV')
plt.scatter(t_idx, ty_idx, s=50, marker = 'v', color = "k", label = 'Peak Min: ' + str(round(ty_idx, 2)) + ' uV')


plt.fill_between(x, (res-res_std), (res+res_std), facecolor = 'green', alpha = 0.18)

amp = y_idx-ty_idx
print('Peak to Peak Amplitude: ' + str(round(amp,2)) + 'uV')
plt.legend(title = ('Amplitude: ' + str(round(amp,2)) + 'uV'))
plt.show()




#%%








#%%

#F-wave w/ 200us
filepath = "C:/Users/Lab/Box/Seanez_Lab/SharedFolders/RAW DATA/Russian_Stim/RS001/RS001_20230605/analog_roi_emg_200us_monophasic_f_1.csv"
columns = ['Trigger', 'Soleus', 'MG']
df_fwave_200 = pd.read_csv(filepath, names = columns)
trigger = df_fwave_200['Trigger']
soleus = (df_fwave_200['Soleus'] / 500 ) * (10**6) #mV
soleus = soleus - np.mean(soleus)

height = np.mean(trigger) + (2*np.std(trigger)) #height requirement for peak detection
peaks, _ = find_peaks(trigger, height=height, distance=500)

#plt.plot(trigger) #plot trigger signal
#plt.plot(peaks, trigger[peaks], "x", markersize = 10) #plot markers on peaks
#plt.axhline(y=height, linestyle = '--', color = 'k')

#%%
#All Data
starting_idx = peaks - 500
ending_idx = peaks + 3000

total_time = 3500/50000 #4400 #Hz, or samples/sec
x = np.linspace(0, total_time*1000, 3500) 

plt.figure(7)
trials = []

for i in range(len(peaks)):
    plt.plot(x, soleus[starting_idx[i]:ending_idx[i]], color ='k', alpha=0.5, linewidth = 0.2)
    plt.ylabel('Microvolts (uV)')
    plt.xlabel('Milliseconds (ms)')
    plt.title('Right Soleus 200us Unfiltered')
    #plt.grid(visible = True)
    trials.append(soleus[starting_idx[i]:ending_idx[i]])

h = np.array(trials)
res = np.average(h, axis=0)

plt.plot(x, res, color = 'green', label = "Avg Trace")
plt.show()


#%% Filtered Fwaves
import scipy

fs = 50000
lowcut = 500
highcut = 10

trials = []

nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
order = 4

starting_idx = peaks + 1400
ending_idx = peaks + 3200

total_time = 1800/50000 #4400 #Hz, or samples/sec
x = np.linspace(0, total_time*1000, 1800) 

plt.figure(8)
trials = []
fs = 50000  # Sampling frequency
fc = 500  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency

fc2 = 10  # Cut-off frequency of the filter
w2 = fc2 / (fs / 2) # Normalize the frequency


##BANDPASS ENTIRE SIGNAL
sidx = peaks - 500
eidx = peaks + 3000
total_timex = 3500/50000 #4400 #Hz, or samples/sec
xidx = np.linspace(0, total_timex*1000, 3500) 
for i in range(len(peaks)):
    b, a = scipy.signal.butter(3, [high, low], 'band')
    filteredBandPass = scipy.signal.lfilter(b, a, soleus)
    plt.plot(xidx, filteredBandPass[sidx[i]:eidx[i]])
    plt.ylabel('Microvolts (uV)')
    plt.xlabel('Milliseconds (ms)')
    plt.title('Entire Signal Bandpass Filtered')
plt.show()
plt.figure(9)


#BANDPASS ONLY FWAVE SECTION
for i in range(len(peaks)):
    #y = butter_bandpass_filter(soleus[starting_idx[i]:ending_idx[i]], lowcut, highcut, fs, order=6)
    #plt.plot(x, y, label='Filtered signal (Hz)')
    
    b, a = scipy.signal.butter(3, [high, low], 'band')
    filteredBandPass = scipy.signal.lfilter(b, a, soleus[starting_idx[i]:ending_idx[i]])
    plt.plot(x, filteredBandPass, color ='k', alpha=0.6, linewidth = 0.2)

    #b, a = signal.butter(5, w, 'low')
    #output = signal.filtfilt(b, a, (soleus[starting_idx[i]:ending_idx[i]]))
    
    #b, a = signal.butter(5, w2, 'high')
    #output2 = signal.filtfilt(b, a, output)
    
    #plt.plot(x, output2, color ='k', alpha=0.6, linewidth = 0.2)

    #plt.plot(x, soleus[starting_idx[i]:ending_idx[i]], color ='k', alpha=0.7, linewidth = 0.2)
    plt.ylabel('Microvolts (uV)')
    plt.xlabel('Milliseconds (ms)')
    plt.title('Only Right Soleus 200us Bandpass Filtered')
    #plt.grid(visible = True)
    trials.append(soleus[starting_idx[i]:ending_idx[i]])

h = np.array(trials)
res = np.average(h, axis=0)

b, a = scipy.signal.butter(3, [high, low], 'band')
filteredBandPass1 = scipy.signal.lfilter(b, a, res)
plt.plot(x, filteredBandPass1, color ='green', alpha=0.88)

plt.show()


plt.figure(10)
for i in range(len(peaks)):
    #y = butter_bandpass_filter(soleus[starting_idx[i]:ending_idx[i]], lowcut, highcut, fs, order=6)
    #plt.plot(x, y, label='Filtered signal (Hz)')
    
    b, a = scipy.signal.butter(3, [high, low], 'band')
    filteredBandPass = scipy.signal.lfilter(b, a, soleus[starting_idx[i]:ending_idx[i]])
    plt.plot(x, filteredBandPass-filteredBandPass1, color ='k', alpha=0.6, linewidth = 0.2)
plt.show()

#%%






























