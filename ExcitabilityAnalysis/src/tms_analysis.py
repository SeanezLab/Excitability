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
rTApre = rTApre[4000000:]  
rTApost = rTApost[2800000:] 

triggerpre = triggerpre[4000000:] * -1
triggerpost = triggerpost[2800000:] * -1


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
tms_pre = []
ax.set_yticks([])

for i in range(len(peaks)):
    plt.plot(x,(rTApre[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    tms_pre.append((rTApre[starting_idx[i]:ending_idx[i]])+offset[i])


plt.title('Raw MEP Traces EX001 pre')
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

for i in range(len(tms_pre)):
    cb = tms_pre[i]
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
        
    plt.figure(5)
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
            


    plt.title("EX001 Pre MEP Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

print("_______________________________________")
print("EX001 MEP Amplitude Pre Test")
amp_df_pre = pd.DataFrame(amplitude_pre, columns=['Amp 1', 'Amp 2', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()
print("MEP Amp", round(means_pre[0], 2), " +/- ", round(stds_pre[0], 2), "uV")
print("MEP Amp", round(means_pre[1], 2), " +/- ", round(stds_pre[1], 2), "uV")
print("MEP Latency", round(means_pre[2], 2), " +/- ", round(stds_pre[2], 2), "ms")





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
tms_post = []
ax.set_yticks([])

for i in range(len(peaks_post)):
    plt.plot(x,(rTApost[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    tms_post.append((rTApost[starting_idx[i]:ending_idx[i]])+offset[i])


plt.title('Raw MEP Traces EX001 post')
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

for i in range(len(tms_post)):
    cb = tms_post[i]
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
        
    plt.figure(6)
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
            


    plt.title("EX001 POST MEP Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

print("_______________________________________")
print("EX001 MEP Amplitude POST Test")
amp_df_pre = pd.DataFrame(amplitude_pre, columns=['Amp 1', 'Amp 2', 'Time'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()
print("MEP Amp", round(means_pre[0], 2), " +/- ", round(stds_pre[0], 2), "uV")
print("MEP Amp", round(means_pre[1], 2), " +/- ", round(stds_pre[1], 2), "uV")
print("MEP Latency", round(means_pre[2], 2), " +/- ", round(stds_pre[2], 2), "ms")

