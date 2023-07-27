# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:27:46 2023

@author: Lab
"""

from preprocess_csv import *
from pathlib import Path


root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"
fwave_list, tms_list = preprocess_main(root)

#%%
#create class to create and save all MEP metrics and ploting functions too
#create all the individual fwave dataframes
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

tmspre = tms_list[1]
tmspost = tms_list[0]

rTApre = np.array(tmspre['R Tibialis Anterior'])
rTApost = np.array(tmspost['R Tibialis Anterior'])

triggerpre = np.array(tmspre['Trigger'])
triggerpost = np.array(tmspost['Trigger'])

#%%
rTApre = rTApre[3000000:]  
rTApost = rTApost[1600000:] 

triggerpre = triggerpre[3000000:] * -1
triggerpost = triggerpost[1600000:] * -1

#%%

height = np.mean(triggerpre) + (3*np.std(triggerpre[0:2000])) #height requirement for peak detection
peaks, _ = find_peaks(triggerpre, height=height, distance=500)

plt.plot(triggerpre) #plot trigger signal
plt.plot(peaks, triggerpre[peaks], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height, linestyle = '--', color = 'k')


#%%
height_post = np.mean(triggerpost) + (10*np.std(triggerpost[0:2000])) #height requirement for peak detection
peaks_post, _ = find_peaks(triggerpost, height=height_post, distance=500)

plt.plot(triggerpost) #plot trigger signal
plt.plot(peaks_post, triggerpost[peaks_post], "x", markersize = 10) #plot markers on peaks
plt.axhline(y=height_post, linestyle = '--', color = 'k')

#%%#%% EX001 MEP basic plot

starting_idx = peaks
ending_idx = peaks + 350

total_time = 350/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 350)  

offset = np.linspace(0,0.0,60)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))

sns.set()
sns.set_style("white")
tms_pre_EX002 = []
ax.set_yticks([])

for i in range(len(peaks)):
    plt.plot(x,(rTApre[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    tms_pre_EX002.append((rTApre[starting_idx[i]:ending_idx[i]])+offset[i])


plt.title('Raw MEP Traces EX002 pre')
plt.ylabel('V')
plt.xlabel('ms')
plt.show()


#%% EX001 MEP Identification
total_time = 350/4400 
time = np.linspace(0, total_time*1000, 350) 
idx_time = int(10*1000/4400)

listofall = []
offset = np.linspace(0,0.015,60)

fs = 4400
amplitude_pre = []
previous_list = []
post_list = []
max_idx_list = []

for i in range(len(tms_pre_EX002)):
    cb = tms_pre_EX002[i]
    #b, a = signal.butter(4, 100, 'high', fs = fs)
    #y = filtfilt(b, a, cb)

    #cb = y + offset[i]
    cb = cb + offset[i]
    
    
    # rest_period = np.average(cb[0:20])
    # print("rest", rest_period)
    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(10)
    plt.plot(time, cb , color="gray", linewidth=0.8, alpha=0.2)
    #plt.plot(higherPeaks[:,0], higherPeaks[:,1], 'ro')
    #plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')
    
    peak_idx = list(higherPeaks[:,0])
    valley_idx = list(lowerPeaks[:,0])
    #print(valley_idx)
    for b in range(len(peak_idx)):
        #int_idx = int(valley_idx[b])
        int_idx_peaks = int(peak_idx[b])
        #plt.plot(time[int_idx], cb[int_idx], "ko")
        #plt.plot(time[int_idx_peaks], cb[int_idx_peaks], "ko")

    
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
    print(max_idx_list)
    
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
        #plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 10, alpha = 0.5)
        #plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 10, alpha = 0.5)
        #plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_pre.append([value_pre, value_post, time_max])
        
        #true_idx
        onset_line = true_idx[i]
        end_line = endddd[i]
        # #test = othertest[i]
        plt.plot(time[onset_line], cb[onset_line], "*", color = 'pink', markersize= 10)
        plt.plot(time[end_line], cb[end_line], "*", color = 'k', markersize= 10)
        # #plt.plot(time[test], cb[test], "*", color = 'k', markersize= 10)

        plt.plot(time[onset_line:end_line], cb[onset_line:end_line],color = "green")
        #plt.plot(time[max_index:max_index+80], cb[max_index:max_index+80],color = "green")



        # #plt.plot(time[max_index:(post_max_index+66)], cb[max_index:(post_max_index+66)],color = "blue", alpha = 0.5)
        
        #plt.plot(time[prev_3], cb[prev_3], "*", color = 'gray', markersize= 10)
        #plt.plot(time[prev_prev], cb[prev_prev], "*", color = 'k', markersize= 10)
        #plt.plot(time[post_post], cb[post_post], "*", color = 'red', markersize= 10)
        
        

            


    plt.title("EX002 Pre MEP Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

print("_______________________________________")
print("EX002 MEP Amplitude Pre Test")
amp_df_pre = pd.DataFrame(amplitude_pre, columns=['Amp 1', 'Amp 2', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()
print("MEP Amp", round(means_pre[0], 2), " +/- ", round(stds_pre[0], 2), "uV")
print("MEP Amp", round(means_pre[1], 2), " +/- ", round(stds_pre[1], 2), "uV")
print("MEP Latency", round(means_pre[2], 2), " +/- ", round(stds_pre[2], 2), "ms")

#%%EX002 PRE DERIVATIVE

time = np.linspace(0, total_time*1000, 350) 

time2 = np.linspace(0, total_time*1000, 349) 

true_idx = []
endddd = []
othertest = []

for i in range(len(tms_pre_EX002)):
    cb = tms_pre_EX002[i]
    idx_start = max_idx_list[i]-80
    print(max_idx_list)
    idx_end = max_idx_list[i]
    y_diff = cb[idx_start:idx_end]
    total_time = len(y_diff)/4400 
    time = np.linspace(0, total_time*1000, len(y_diff)) 
    numpyDiff = np.diff(y_diff)/np.diff(time)
    
    
    idx_end_post = max_idx_list[i]+80
    idx_start_post = max_idx_list[i]
    y_post = cb[idx_start_post:idx_end_post]
    
    flipped = np.flip(y_post)
    
    total_time_post = len(y_post)/4400 
    time_post = np.linspace(0, total_time_post*1000, len(y_post))
    
    numpyDiff_post = np.diff(flipped)/np.diff(time_post)
    
    
    


    idx3 = np.where(abs(numpyDiff_post)>0.00001)[0][0]
    endddd.append(idx_end_post-idx3)
    plt.figure(25)
    time_post = np.linspace(0, total_time_post*1000, len(y_post)-1) 
    plt.plot(time_post,numpyDiff_post, color = "gray", alpha = 0.5)
    plt.plot(time_post[idx3], numpyDiff_post[idx3], "x", color="red")

    
    
    plt.figure(22)
    time = np.linspace(0, total_time*1000, len(y_diff)-1) 
    plt.plot(time,numpyDiff, color = "gray", alpha = 0.5)    
    idx2 = np.where(abs(numpyDiff)>0.00001)[0][0]
    plt.plot(time[idx2], numpyDiff[idx2], "x", color="red")
    true_idx.append(idx_start+idx2)
    



#%%#%% EX001 MEP basic plot POST
starting_idx = peaks_post
ending_idx = peaks_post + 350

total_time = 350/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 350)  

offset = np.linspace(0,0.0,60)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))

sns.set()
sns.set_style("white")
tms_post_EX002 = []
ax.set_yticks([])

for i in range(len(peaks_post)):
    plt.plot(x,(rTApost[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    tms_post_EX002.append((rTApost[starting_idx[i]:ending_idx[i]])+offset[i])


plt.title('Raw MEP Traces EX002 post')
plt.ylabel('V')
plt.xlabel('ms')
plt.show()


#%% EX001 MEP Identification
total_time = 350/4400 
time = np.linspace(0, total_time*1000, 350) 

listofall = []
offset = np.linspace(0,0.015,60)

fs = 4400
amplitude_pre = []

for i in range(len(tms_post_EX002)):
    cb = tms_post_EX002[i]
    #b, a = signal.butter(4, 100, 'high', fs = fs)
    #y = filtfilt(b, a, cb)

    #cb = y + offset[i]
    cb = cb + offset[i]
    
    
    # rest_period = np.average(cb[0:20])
    # print("rest", rest_period)
    peaks = peakdetect(cb, lookahead=10)
        
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    
       
    if len(higherPeaks) == 0:
        higherPeaks = np.zeros((2,2))
    if len(lowerPeaks) == 0:
        lowerPeaks = np.zeros((2,2))
        
    plt.figure(8)
    plt.plot(time, cb , color="gray", linewidth=0.8, alpha=0.8)
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

    value_pre = abs((cb[max_index]-cb[previous_max_index]) * (10**6))
    value_post =  abs((cb[max_index]-cb[post_max_index]) * (10**6))
    time_max = time[[max_index]]
    
    
    if value_pre >= 50 or value_post >= 50:

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 5, alpha = 0.5)
        plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 5)
        plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_pre.append([value_pre, value_post, time_max])
            


    plt.title("EX002 POST MEP Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

print("_______________________________________")
print("EX002 MEP Amplitude POST Test")
amp_df_pre = pd.DataFrame(amplitude_pre, columns=['Amp 1', 'Amp 2', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()
print("MEP Amp", round(means_pre[0], 2), " +/- ", round(stds_pre[0], 2), "uV")
print("MEP Amp", round(means_pre[1], 2), " +/- ", round(stds_pre[1], 2), "uV")
print("MEP Latency", round(means_pre[2], 2), " +/- ", round(stds_pre[2], 2), "ms")

