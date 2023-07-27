# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:58:33 2023

@author: Lab
"""
from preprocess_csv import *
from pathlib import Path

root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"
fwave_list, tms_list = preprocess_main(root)

#%%
#create all the individual fwave dataframes
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fwavepre = fwave_list[1]
fwavepost = fwave_list[0]

rTApre = np.array(fwavepre['R Tibialis Anterior'])
rTApost = np.array(fwavepost['R Tibialis Anterior'])

triggerpre = np.array(fwavepre['Trigger'])
triggerpost = np.array(fwavepost['Trigger'])

#%%EX001 specifics
rTApre = rTApre[1600000:]  
rTApost = rTApost[260000:] 

triggerpre = triggerpre[1600000:] * -1
triggerpost = triggerpost[260000:] * -1


height = np.mean(triggerpre) + (3*np.std(triggerpre[0:20000])) #height requirement for peak detection
peaks, _ = find_peaks(triggerpre, height=height, distance=500)

# #plt.plot(triggerpre) #plot trigger signal
# #plt.plot(peaks, triggerpre[peaks], "x", markersize = 10) #plot markers on peaks
# #plt.axhline(y=height, linestyle = '--', color = 'k')
#%%

starting_idx = peaks - 50
ending_idx = peaks + 250

total_time = 300/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 300)  
trigger_idx = x[49]

offset = np.linspace(0,0.0,60)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))

sns.set()
sns.set_style("white")
trials_pre_EX001 = []
ax.set_yticks([])


for i in range(len(peaks)):
    plt.plot(x,(rTApre[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre_EX001.append((rTApre[starting_idx[i]:ending_idx[i]])+offset[i])


plt.axvline(33, linestyle = "--", color = 'gray')
plt.axvline(40, linestyle = "--", color = 'gray')
plt.axvline(65, linestyle = "--", color = 'gray')


plt.title('Raw F-wave Traces EX001 pre')
plt.ylabel('V')
plt.xlabel('ms')
plt.show()

ms = 33
to_index = int(ms/1000*440)
idx = 49
to_ms = int(idx/4400*1000)

plt.axvline(to_ms, linestyle = "--", color = 'gray')



#%%#FWAVE POST EX001

height = np.mean(triggerpost) + (10*np.std(triggerpost[0:5000])) #height requirement for peak detection
peaks, _ = find_peaks(triggerpost, height=height, distance=500)

#plt.plot(triggerpost) #plot trigger signal
#plt.plot(peaks, triggerpost[peaks], "x", markersize = 10) #plot markers on peaks
#plt.axhline(y=height, linestyle = '--', color = 'k')

starting_idx = peaks - 50
ending_idx = peaks + 250

total_time = 300/4400 #15.9091 ms
x = np.linspace(0, total_time*1000, 300)  

offset = np.linspace(0,0.0,60)
#offset = np.linspace(0,0.004,60)

import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks([])

sns.set()
sns.set_style("white")

trials_post_EX001 = []
for i in range(len(peaks)):
    plt.plot(x,(rTApost[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_post_EX001.append((rTApost[starting_idx[i]:ending_idx[i]])+offset[i])


plt.axvline(33, linestyle = "--", color = 'gray')
plt.axvline(40, linestyle = "--", color = 'gray')
plt.axvline(65, linestyle = "--", color = 'gray')

plt.axvline(to_ms, linestyle = "--", color = 'gray')



plt.title('Raw F-wave Traces EX001 post')
plt.ylabel('V')
plt.xlabel('ms')
plt.show()
