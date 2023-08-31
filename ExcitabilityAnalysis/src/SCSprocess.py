# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:33:39 2023

@author: marie
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print("FWAVE PRE: Read in pre dataframe and muscle specifics")

#in future call from fwave_names list or tms_names list
df1 = df_SCS.loc['EX001_20230721_F_Waves_fwavepre_SCS']
#pull RTA and trigger data
RTA = np.array(df1['R Tibialis Anterior'])
TRIG = np.array(df1['Trigger'])

#%%
RTA = RTA[30000:]
TRIG = TRIG[30000:] * -1

height = np.mean(TRIG) + (3*np.std(TRIG[400000:])) #height requirement for peak detection
peaks, _ = find_peaks(TRIG, height=height, distance=500)

print("Plot trigger peak identification")
plt.plot(TRIG) #plot trigger signal
plt.plot(peaks, TRIG[peaks], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height, linestyle = '--', color = 'k')

#%%
print("FWAVE PRE: Get ROI of RTA from TRIG peaks")

starting_idx = peaks - 50
ending_idx = peaks + 250

total_time = 300/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 300)  

#offset = np.linspace(0,0.01,60)
offset = np.linspace(0,0.0,60)

fig, ax = plt.subplots(figsize=(7,12))

trials_pre = []

for i in range(len(peaks)):
    plt.plot(x,(RTA[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre.append((RTA[starting_idx[i]:ending_idx[i]])+offset[i]) 
    
plt.title("All Traces EX001 Fwave Pre SCS")
plt.xlabel("ms")
plt.ylabel("V")
plt.savefig("EX001_FWaves_PRE_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


#%%
from peakdetect import peakdetect
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd

print("FWAVE PRE: Get Fwave amplitude and persistance")

total_time = 140/4400 
time = np.linspace(0, total_time*1000, 140) 

listofall = []
offset = np.linspace(0,0.0012,60)

fs = 4400
amplitude_pre = []
three_idxs = []

for i in range(len(trials_pre)):
    cb = trials_pre[i]
    cb = cb[160:]
    b, a = signal.butter(4, 100, 'high', fs = fs)
    y = filtfilt(b, a, cb)

    cb = y + offset[i]
    
    
    # rest_period = np.average(cb[0:20])
    # print("rest", rest_period)
    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(3, figsize=(7,12))
    plt.plot(time, cb, color="gray", linewidth=0.8, alpha=0.8)
    #plt.plot(higherPeaks[:,0], higherPeaks[:,1], 'ro')
    #plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')
    
    peak_idx = list(higherPeaks[:,0])
    valley_idx = list(lowerPeaks[:,0])
    
    merged = []
    while peak_idx or valley_idx:
        if peak_idx:
            merged.append(peak_idx.pop())
        if valley_idx:
            merged.append(valley_idx.pop(0))
    #print("do you work", sorted(merged))
    merged = sorted(merged)    
    listofall.append(sorted(merged))
    
    #print("i:", i, "Merged:", merged)
    save_idx = []
    values_and_idx = []
    for k in range(len(merged)):
        values_and_idx.append([int(merged[k]), cb[int(merged[k])]])
    df_vi = pd.DataFrame(values_and_idx, columns=['idx_key', 'idx_value'])
    #print(df_vi)
    #for l in range(len(merged)):
    index_max = df_vi.idx_value.argmax()
    
    max_index = df_vi.idx_key.iloc[index_max]
    previous_max_index = df_vi.idx_key.iloc[index_max-1]
    
    if index_max+1 == len(df_vi.index):
        post_max_index = 0
        print("Do nothing")
    else:
        post_max_index = df_vi.idx_key.iloc[index_max+1]

    value_pre = abs((cb[max_index]-cb[previous_max_index]) * (10**6)) #converts to uV
    value_post =  abs((cb[max_index]-cb[post_max_index]) * (10**6)) #converts to uV
    
    
    if value_pre >= 20 or value_post >= 20: #uV

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 5, alpha = 0.5)
        plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 5)
        plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_pre.append(max(value_pre, value_post))
        three_idxs.append([previous_max_index, max_index, post_max_index])
        
         

    plt.title("EX001 Pre Fwave Identification: SCS")
    plt.xlabel("ms")
    plt.ylabel("V")
    
    ax = plt.gca()
    
    # Hide X and Y axes label marks
    ax.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    ax.set_yticks([])

persistance_pre = (len(amplitude_pre))/60 * 100
print("_______________________________________")
print("EX001 Fwave Persisstance and Amplitude Pre Test for SCS")
#print("Persistance: ", persistance_pre, "%")
amp_df_pre = pd.DataFrame(amplitude_pre)
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()


amp = "Fwave Amp: " + str(round(means_pre[0], 2)) + " +/- "+ str(round(stds_pre[0], 2)) + "uV"
print(amp)
pers = "Persistance: " + str(round(persistance_pre, 2)) + "%"
print(pers)

ax.text(23, 0.001, amp + "\n" + pers, bbox=dict(facecolor='white'))
plt.savefig("EX001_20230721_FWaves_PRE_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


#%%











#%%

print("FWAVE POST: Read in pre dataframe and muscle specifics")

#in future call from fwave_names list or tms_names list
df1 = df_SCS.loc['EX001_20230721_F_Waves_fwavepost_SCS']
#pull RTA and trigger data
RTA = np.array(df1['R Tibialis Anterior'])
TRIG = np.array(df1['Trigger'])

#%%
RTA = RTA[10000:]
TRIG = TRIG[10000:] * -1

height = np.mean(TRIG) + (3*np.std(TRIG[400000:])) #height requirement for peak detection
peaks, _ = find_peaks(TRIG, height=height, distance=500)

print("Plot trigger peak identification")
plt.plot(TRIG) #plot trigger signal
plt.plot(peaks, TRIG[peaks], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height, linestyle = '--', color = 'k')

#%%
print("FWAVE POST: Get ROI of RTA from TRIG peaks")

starting_idx = peaks - 50
ending_idx = peaks + 250

total_time = 300/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 300)  

#offset = np.linspace(0,0.01,60)
offset = np.linspace(0,0.0,60)

fig, ax = plt.subplots(figsize=(7,12))

trials_pre = []

for i in range(len(peaks)):
    plt.plot(x,(RTA[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre.append((RTA[starting_idx[i]:ending_idx[i]])+offset[i])
    
plt.title("All Traces EX001 Fwave POST SCS")
plt.xlabel("ms")
plt.ylabel("V")
plt.savefig("EX001_FWaves_POST_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


#%%
from peakdetect import peakdetect
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd

print("FWAVE POST: Get Fwave amplitude and persistance")

total_time = 140/4400 
time = np.linspace(0, total_time*1000, 140) 

listofall = []
offset = np.linspace(0,0.0012,60)

fs = 4400
amplitude_pre = []
three_idxs = []

for i in range(len(trials_pre)):
    cb = trials_pre[i]
    cb = cb[160:]
    b, a = signal.butter(4, 100, 'high', fs = fs)
    y = filtfilt(b, a, cb)

    cb = y + offset[i]
    
    
    
    # rest_period = np.average(cb[0:20])
    # print("rest", rest_period)
    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(3, figsize=(7,12))
    plt.plot(time, cb, color="gray", linewidth=0.8, alpha=0.8)
    #plt.plot(higherPeaks[:,0], higherPeaks[:,1], 'ro')
    #plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')
    
    peak_idx = list(higherPeaks[:,0])
    valley_idx = list(lowerPeaks[:,0])
    
    merged = []
    while peak_idx or valley_idx:
        if peak_idx:
            merged.append(peak_idx.pop())
        if valley_idx:
            merged.append(valley_idx.pop(0))
    #print("do you work", sorted(merged))
    merged = sorted(merged)    
    listofall.append(sorted(merged))
    
    #print("i:", i, "Merged:", merged)
    save_idx = []
    values_and_idx = []
    for k in range(len(merged)):
        values_and_idx.append([int(merged[k]), cb[int(merged[k])]])
    df_vi = pd.DataFrame(values_and_idx, columns=['idx_key', 'idx_value'])
    #print(df_vi)
    #for l in range(len(merged)):
    index_max = df_vi.idx_value.argmax()
    
    max_index = df_vi.idx_key.iloc[index_max]
    previous_max_index = df_vi.idx_key.iloc[index_max-1]
    
    if index_max+1 == len(df_vi.index):
        post_max_index = 0
        print("Do nothing")
    else:
        post_max_index = df_vi.idx_key.iloc[index_max+1]

    value_pre = abs((cb[max_index]-cb[previous_max_index]) * (10**6)) #converts to uV
    value_post =  abs((cb[max_index]-cb[post_max_index]) * (10**6)) #converts to uV
    
    
    if value_pre >= 20 or value_post >= 20: #uV

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 5, alpha = 0.5)
        plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 5)
        plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_pre.append(max(value_pre, value_post))
        three_idxs.append([previous_max_index, max_index, post_max_index])
        
         

    plt.title("EX001 POST Fwave Identification: SCS")
    plt.xlabel("ms")
    plt.ylabel("V")
    
    ax = plt.gca()
    
    # Hide X and Y axes label marks
    ax.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    ax.set_yticks([])

persistance_pre = (len(amplitude_pre))/60 * 100
print("_______________________________________")
print("EX001 Fwave Persisstance and Amplitude POST Test for SCS")
#print("Persistance: ", persistance_pre, "%")
amp_df_pre = pd.DataFrame(amplitude_pre)
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()


amp = "Fwave Amp: " + str(round(means_pre[0], 2)) + " +/- "+ str(round(stds_pre[0], 2)) + "uV"
print(amp)
pers = "Persistance: " + str(round(persistance_pre, 2)) + "%"
print(pers)

ax.text(23, 0.001, amp + "\n" + pers, bbox=dict(facecolor='white'))
plt.savefig("EX001_20230721_FWaves_POST_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')

#%%











#%%
print("TMS pre: Read in pre dataframe and muscle specifics")

#in future call from fwave_names list or tms_names list
df1 = df_SCS.loc['EX001_20230721_TMS_tmspre_SCS']
#pull RTA and trigger data
RTA = np.array(df1['R Tibialis Anterior'])
TRIG = np.array(df1['Trigger'])

#%%
#RTA = RTA[2500000:]
TRIG = TRIG * -1

height = np.mean(TRIG) + (3*np.std(TRIG[:1000000])) #height requirement for peak detection
peaks, _ = find_peaks(TRIG, height=height, distance=500)

print("Plot trigger peak identification")
plt.plot(TRIG) #plot trigger signal
plt.plot(peaks, TRIG[peaks], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height, linestyle = '--', color = 'k')
#%%
print("TMS pre: Get ROI of RTA from TRIG peaks")

starting_idx = peaks
ending_idx = peaks + 350

total_time = 350/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 350)  

#offset = np.linspace(0,0.01,60)
offset = np.linspace(0,0.0,60)

fig, ax = plt.subplots(figsize=(7,12))

trials_pre = []

for i in range(len(peaks)):
    plt.plot(x,(RTA[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre.append((RTA[starting_idx[i]:ending_idx[i]])+offset[i])
    
plt.title("All Traces EX001 MEP PRE SCS")
plt.xlabel("ms")
plt.ylabel("V")
plt.savefig("EX001_TMS_PRE_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


    
#%% EX001 MEP Identification
import matplotlib.pyplot as plt

total_time = 350/4400 
time = np.linspace(0, total_time*1000, 350) 

listofall = []

fs = 4400
amplitude_pre = []
previous_list = []
post_list = []
max_idx_list = []
MEPS = []
offset = np.linspace(0,0.002,30)

fig, ax = plt.subplots(figsize=(7,12))


for i in range(len(trials_pre)):
    cb = trials_pre[i]
    cb = cb + offset[i]

    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(1)
    plt.plot(time, cb , color="gray", linewidth=0.8, alpha=0.7)
    #plt.plot(higherPeaks[:,0], higherPeaks[:,1], 'ro')
    #plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')
    
    peak_idx = list(higherPeaks[:,0])
    valley_idx = list(lowerPeaks[:,0])
    for b in range(len(peak_idx)):
        int_idx_peaks = int(peak_idx[b])

    merged = []
    while peak_idx or valley_idx:
        if peak_idx:
            merged.append(peak_idx.pop())
        if valley_idx:
            merged.append(valley_idx.pop(0))
    #print("do you work", sorted(merged))
    merged = sorted(merged)    
    listofall.append(sorted(merged))
    
    #print("i:", i, "Merged:", merged)
    values_and_idx = []
    for k in range(len(merged)):
        values_and_idx.append([int(merged[k]), cb[int(merged[k])]])
    df_vi = pd.DataFrame(values_and_idx, columns=['idx_key', 'idx_value'])
    #print(df_vi)
    #for l in range(len(merged)):
    index_max = df_vi.idx_value.argmax()
    
    max_index = df_vi.idx_key.iloc[index_max]
    max_idx_list.append(max_index)
    #print(max_idx_list)
    
    previous_max_index = df_vi.idx_key.iloc[index_max-1]
    prev_prev = df_vi.idx_key.iloc[index_max-2]
    prev_3 = df_vi.idx_key.iloc[index_max-3]
    previous_list.append(previous_max_index)
    
    if index_max+1 == len(df_vi.index):
        post_max_index = 0
        print("Do nothing")
        post_list.append(post_max_index)
    else:
        post_max_index = df_vi.idx_key.iloc[index_max+1]
        post_list.append(post_max_index)
        
    if index_max+2 == len(df_vi.index):
        post_post = 0
        print("Do nothing")
    else:
        post_post = df_vi.idx_key.iloc[index_max+2]


    value_pre = abs((cb[max_index]-cb[previous_max_index]) * (10**6))
    value_post =  abs((cb[max_index]-cb[post_max_index]) * (10**6))
    time_max = time[[max_index]]
    
    
    if value_pre >= 50 or value_post >= 50:

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        
        onset_line = true_idx[i]
        end_line = endddd[i]
        plt.plot(time[onset_line], cb[onset_line], "*", color = 'green', markersize= 10)
        plt.plot(time[end_line], cb[end_line], "*", color = 'blue', markersize= 10)

        plt.plot(time[onset_line:end_line], cb[onset_line:end_line],color = "black")
        
        max_value = cb[max_index]
        min_value = min(np.array(cb[onset_line:end_line]))
        mep = ((max_value-min_value) * (10**6))
        latency = time[onset_line]
        #print("AMPS MEP", (max_value-min_value) * (10**6))
        amplitude_pre.append([mep, latency])

    

    plt.title("EX001 Pre MEP Identification: SCS")
    plt.xlabel("ms")
    plt.ylabel("V")
    ax = plt.gca()
    
    # Hide X and Y axes label marks
    ax.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    ax.set_yticks([])
    
    #plt.show()

print("_______________________________________")
print("EX001 MEP Amplitude Pre Test: SCS")

amp_df_pre = pd.DataFrame(amplitude_pre, columns=['MEP', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()

amp = "Amp: "+str(round(means_pre[0], 2))+" +/- "+str(round(stds_pre[0], 2))+"uV"
print(amp)
pers = "Latency: "+str(round(means_pre[1], 2))+" +/- "+str(round(stds_pre[1], 2))+"ms"
print(pers)

ax.text(75, 0.0017, amp + "\n" + pers, bbox=dict(facecolor='white'))

plt.savefig("EX001_20230721_TMS_PRE_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


#%%EX001 PRE DERIVATIVE

time = np.linspace(0, total_time*1000, 350) 

time2 = np.linspace(0, total_time*1000, 349) 

true_idx = []
endddd = []
othertest = []

for i in range(len(trials_pre)):
    cb = trials_pre[i]
    idx_start = 10
    #idx_start = max_idx_list[i]-110
   # print(max_idx_list)
    idx_end = max_idx_list[i]
    y_diff = cb[idx_start:idx_end]
    total_time = len(y_diff)/4400 
    time = np.linspace(0, total_time*1000, len(y_diff)) 
    numpyDiff = np.diff(y_diff)/np.diff(time)
    
    
    #idx_end_post = max_idx_list[i]+110
    idx_end_post = 330
    idx_start_post = max_idx_list[i]
    y_post = cb[idx_start_post:idx_end_post]
    
    flipped = np.flip(y_post)
    
    total_time_post = len(y_post)/4400 
    time_post = np.linspace(0, total_time_post*1000, len(y_post))
    
    numpyDiff_post = np.diff(flipped)/np.diff(time_post)
    
    idx3 = np.where(abs(numpyDiff_post)>0.00002)[0][0]
    endddd.append(idx_end_post-idx3)
    plt.figure(25)
    time_post = np.linspace(0, total_time_post*1000, len(y_post)-1) 
    plt.plot(time_post,numpyDiff_post, color = "gray", alpha = 0.5)
    plt.plot(time_post[idx3], numpyDiff_post[idx3], "x", color="red")


    plt.figure(22)
    time = np.linspace(0, total_time*1000, len(y_diff)-1) 
    plt.plot(time,numpyDiff, color = "gray", alpha = 0.5)    
    idx2 = np.where(abs(numpyDiff)>0.00002)[0][0]
    plt.plot(time[idx2], numpyDiff[idx2], "x", color="red")
    true_idx.append(idx_start+idx2) 

#%%












#%%
print("TMS POST: Read in pre dataframe and muscle specifics")

#in future call from fwave_names list or tms_names list
df1 = df_SCS.loc['EX001_20230721_TMS_tmspost_SCS']
#pull RTA and trigger data
RTA = np.array(df1['R Tibialis Anterior'])
TRIG = np.array(df1['Trigger'])

#%%
RTA = RTA[1800000:]
TRIG = TRIG[1800000:] * -1

height = np.mean(TRIG) + (3*np.std(TRIG[:200000])) #height requirement for peak detection
peaks, _ = find_peaks(TRIG, height=height, distance=500)

print("Plot trigger peak identification")
plt.plot(TRIG) #plot trigger signal
plt.plot(peaks, TRIG[peaks], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height, linestyle = '--', color = 'k')
#%%
print("TMS POST: Get ROI of RTA from TRIG peaks")

starting_idx = peaks
ending_idx = peaks + 350

total_time = 350/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 350)  

#offset = np.linspace(0,0.01,60)
offset = np.linspace(0,0.0,60)

fig, ax = plt.subplots(figsize=(7,12))

trials_pre = []

for i in range(len(peaks)):
    plt.plot(x,(RTA[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre.append((RTA[starting_idx[i]:ending_idx[i]])+offset[i])
    
plt.title("All Traces EX001 MEP POST SCS")
plt.xlabel("ms")
plt.ylabel("V")
plt.savefig("EX001_TMS_POST_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


    
    
#%% EX001 MEP Identification
import matplotlib.pyplot as plt

total_time = 350/4400 
time = np.linspace(0, total_time*1000, 350) 

listofall = []

fs = 4400
amplitude_pre = []
previous_list = []
post_list = []
max_idx_list = []
MEPS = []
offset = np.linspace(0,0.002,30)
fig, ax = plt.subplots(figsize=(7,12))


for i in range(len(trials_pre)):
    cb = trials_pre[i]
    cb = cb + offset[i]

    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(6, figsize=(7,12))
    plt.plot(time, cb , color="gray", linewidth=0.8, alpha=0.7)
    #plt.plot(higherPeaks[:,0], higherPeaks[:,1], 'ro')
    #plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')
    
    peak_idx = list(higherPeaks[:,0])
    valley_idx = list(lowerPeaks[:,0])
    for b in range(len(peak_idx)):
        int_idx_peaks = int(peak_idx[b])

    merged = []
    while peak_idx or valley_idx:
        if peak_idx:
            merged.append(peak_idx.pop())
        if valley_idx:
            merged.append(valley_idx.pop(0))
    #print("do you work", sorted(merged))
    merged = sorted(merged)    
    listofall.append(sorted(merged))
    
    #print("i:", i, "Merged:", merged)
    values_and_idx = []
    for k in range(len(merged)):
        values_and_idx.append([int(merged[k]), cb[int(merged[k])]])
    df_vi = pd.DataFrame(values_and_idx, columns=['idx_key', 'idx_value'])
    #print(df_vi)
    #for l in range(len(merged)):
    index_max = df_vi.idx_value.argmax()
    
    max_index = df_vi.idx_key.iloc[index_max]
    max_idx_list.append(max_index)
    #print(max_idx_list)
    
    previous_max_index = df_vi.idx_key.iloc[index_max-1]
    prev_prev = df_vi.idx_key.iloc[index_max-2]
    prev_3 = df_vi.idx_key.iloc[index_max-3]
    previous_list.append(previous_max_index)
    
    if index_max+1 == len(df_vi.index):
        post_max_index = 0
        print("Do nothing")
        post_list.append(post_max_index)
    else:
        post_max_index = df_vi.idx_key.iloc[index_max+1]
        post_list.append(post_max_index)
        
    if index_max+2 == len(df_vi.index):
        post_post = 0
        print("Do nothing")
    else:
        post_post = df_vi.idx_key.iloc[index_max+2]


    value_pre = abs((cb[max_index]-cb[previous_max_index]) * (10**6))
    value_post =  abs((cb[max_index]-cb[post_max_index]) * (10**6))
    time_max = time[[max_index]]
    
    
    if value_pre >= 50 or value_post >= 50:

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        
        onset_line = true_idx[i]
        end_line = endddd[i]
        plt.plot(time[onset_line], cb[onset_line], "*", color = 'green', markersize= 10)
        plt.plot(time[end_line], cb[end_line], "*", color = 'blue', markersize= 10)

        plt.plot(time[onset_line:end_line], cb[onset_line:end_line],color = "black")
        
        max_value = cb[max_index]
        min_value = min(np.array(cb[onset_line:end_line]))
        mep = ((max_value-min_value) * (10**6))
        latency = time[onset_line]
        #print("AMPS MEP", (max_value-min_value) * (10**6))
        amplitude_pre.append([mep, latency])

    

    plt.title("EX001 POST MEP Identification: SCS")
    plt.xlabel("ms")
    plt.ylabel("V")
    ax = plt.gca()
    
    # Hide X and Y axes label marks
    ax.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    ax.set_yticks([])
    
    #plt.show()

print("_______________________________________")
print("EX001 MEP Amplitude Post Test: SCS")

amp_df_pre = pd.DataFrame(amplitude_pre, columns=['MEP', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()

amp = "Amp: "+str(round(means_pre[0], 2))+" +/- "+str(round(stds_pre[0], 2))+"uV"
print(amp)
pers = "Latency: "+str(round(means_pre[1], 2))+" +/- "+str(round(stds_pre[1], 2))+"ms"
print(pers)

ax.text(75, 0.0017, amp + "\n" + pers, bbox=dict(facecolor='white'))

plt.savefig("EX001_20230721_TMS_POST_SCS.png", format="png", dpi=1200, bbox_inches = 'tight')


#%%EX002 PRE DERIVATIVE

time = np.linspace(0, total_time*1000, 350) 

time2 = np.linspace(0, total_time*1000, 349) 

true_idx = []
endddd = []
othertest = []

for i in range(len(trials_pre)):
    cb = trials_pre[i]
    #idx_start = 10
    idx_start = max_idx_list[i]-80
   # print(max_idx_list)
    idx_end = max_idx_list[i]
    y_diff = cb[idx_start:idx_end]
    total_time = len(y_diff)/4400 
    time = np.linspace(0, total_time*1000, len(y_diff)) 
    numpyDiff = np.diff(y_diff)/np.diff(time)
    
    
    idx_end_post = max_idx_list[i]+80
    #idx_end_post = 330
    idx_start_post = max_idx_list[i]
    y_post = cb[idx_start_post:idx_end_post]
    
    flipped = np.flip(y_post)
    
    total_time_post = len(y_post)/4400 
    time_post = np.linspace(0, total_time_post*1000, len(y_post))
    
    numpyDiff_post = np.diff(flipped)/np.diff(time_post)
    
    idx3 = np.where(abs(numpyDiff_post)>0.000013)[0][0]
    endddd.append(idx_end_post-idx3)
    plt.figure(25)
    time_post = np.linspace(0, total_time_post*1000, len(y_post)-1) 
    plt.plot(time_post,numpyDiff_post, color = "gray", alpha = 0.5)
    plt.plot(time_post[idx3], numpyDiff_post[idx3], "x", color="red")


    plt.figure(22)
    time = np.linspace(0, total_time*1000, len(y_diff)-1) 
    plt.plot(time,numpyDiff, color = "gray", alpha = 0.5)    
    idx2 = np.where(abs(numpyDiff)>0.000013)[0][0]
    plt.plot(time[idx2], numpyDiff[idx2], "x", color="red")
    true_idx.append(idx_start+idx2) 