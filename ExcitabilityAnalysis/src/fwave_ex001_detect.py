# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:47:17 2023

@author: Lab
"""


from peakdetect import peakdetect
from scipy import signal
from scipy.signal import butter, filtfilt

#############################################   EX001 PRE

total_time = 150/4400 
time = np.linspace(0, total_time*1000, 150) 
idx_5ms = int(10/1000*4400)

listofall = []
offset = np.linspace(0,0.004,60)

fs = 4400
amplitude_pre = []

for i in range(len(trials_pre_EX001)):
    cb = trials_pre_EX001[i]
    cb = cb[150:]
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
        
    plt.figure(3)
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
    
    
    if value_pre >= 50 or value_post >= 50:

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 5, alpha = 0.5)
        plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 5)
        plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_pre.append([value_pre, value_post])
            


    plt.title("EX001 Pre Fwave Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

persistance_pre = (len(amplitude_pre))/60 * 100
print("_______________________________________")
print("EX001 Fwave Persisstance and Amplitude Pre Test")
print("Persistance: ", persistance_pre, "%")
amp_df_pre = pd.DataFrame(amplitude_pre, columns=['Amp 1', 'Amp 2'])
means_pre = amp_df_pre.mean()
stds_pre = amp_df_pre.std()
print("Fwave Amp", round(means_pre[0], 2), " +/- ", round(stds_pre[0], 2), "uV")
print("Fwave Amp", round(means_pre[1], 2), " +/- ", round(stds_pre[1], 2), "uV")


    
#%%    
#############################################   EX001 POST
listofall = []
offset = np.linspace(0,0.004,60)


amplitude_post = []

for i in range(len(trials_post_EX001)):
    cb = trials_post_EX001[i]
    cb = cb[150:]
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
        
    plt.figure(4)
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
    
    
    if value_pre >= 50 or value_post >= 50:

        plt.plot(time[max_index], cb[max_index], "x", color = "red", markersize = 10)
        plt.plot(time[previous_max_index], cb[previous_max_index], "o", color = "green", markersize = 5, alpha = 0.5)
        plt.plot(time[post_max_index], cb[post_max_index], "o", color = "blue", markersize = 5)
        plt.plot(time[previous_max_index:post_max_index+1], cb[previous_max_index:post_max_index+1], color = "k")
        
        amplitude_post.append([value_pre, value_post])
            
    plt.title("EX001 Post Fwave Identification")
    plt.xlabel("ms")
    plt.ylabel("V")

persistance_post = (len(amplitude_post))/60 * 100
print("_______________________________________")
print("EX001 Fwave Persisstance and Amplitude Post Test")
print("Persistance: ", persistance_post, "%")
amp_df_post = pd.DataFrame(amplitude_post, columns=['Amp 1', 'Amp 2'])
means_post = amp_df_post.mean()
stds_post = amp_df_post.std()
print("Fwave Amp", round(means_post[0], 2), " +/- ", round(stds_post[0], 2), "uV")
print("Fwave Amp", round(means_post[1], 2), " +/- ", round(stds_post[1], 2), "uV")









