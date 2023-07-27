from preprocess_csv import *
from pathlib import Path

root = Path.home() / "Box/Seanez_Lab/SharedFolders/RAW DATA/Excitability"
fwave_list, tms_list = preprocess_main(root)

savepath = "C:/Users/Lab/Documents/ExcitabilityAnalysis/data/processed"
#%%
#create class to save all Fwave metrics and ploting functions too
#class fwave_metrics(fwave_list, savepath):
#    def init(self):
#        self.fwave = fwave_list
#        self.savepath = savepath

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
# rTApre = rTApre[1600000:]  
# rTApost = rTApost[260000:] 

# triggerpre = triggerpre[1600000:] * -1
# triggerpost = triggerpost[260000:] * -1


# height = np.mean(triggerpre) + (3*np.std(triggerpre[0:20000])) #height requirement for peak detection
# peaks, _ = find_peaks(triggerpre, height=height, distance=500)

# #plt.plot(triggerpre) #plot trigger signal
# #plt.plot(peaks, triggerpre[peaks], "x", markersize = 10) #plot markers on peaks
# #plt.axhline(y=height, linestyle = '--', color = 'k')

#EX002 specifics
rTApre = rTApre[400000:750000]  
rTApost = rTApost[50000:] 

triggerpre = triggerpre[400000:750000] * -1
triggerpost = triggerpost[50000:] * -1


height = np.mean(triggerpre) + (3*np.std(triggerpre[0:20000])) #height requirement for peak detection
peaks, _ = find_peaks(triggerpre, height=height, distance=500)

#plt.plot(triggerpre) #plot trigger signal
#plt.plot(peaks, triggerpre[peaks], "x", markersize = 10) #plot markers on peaks
#plt.axhline(y=height, linestyle = '--', color = 'k')

#%%
starting_idx = peaks - 50
ending_idx = peaks + 250

total_time = 300/4400 #68.18 ms
x = np.linspace(0, total_time*1000, 300)  
trigger_idx = x[49]

offset = np.linspace(0,0.01,60)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))

sns.set()
sns.set_style("white")
trials_pre = []
ax.set_yticks([])


for i in range(len(peaks)):
    plt.plot(x,(rTApre[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_pre.append((rTApre[starting_idx[i]:ending_idx[i]])+offset[i])


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

offset = np.linspace(0,0.01,60)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks([])

sns.set()
sns.set_style("white")

trials_post = []
for i in range(len(peaks)):
    plt.plot(x,(rTApost[starting_idx[i]:ending_idx[i]])+offset[i],linewidth = .8,
             color = 'gray', alpha=0.8)
    trials_post.append((rTApost[starting_idx[i]:ending_idx[i]])+offset[i])


plt.axvline(33, linestyle = "--", color = 'gray')
plt.axvline(40, linestyle = "--", color = 'gray')
plt.axvline(65, linestyle = "--", color = 'gray')

plt.axvline(to_ms, linestyle = "--", color = 'gray')



plt.title('Raw F-wave Traces EX001 post')
plt.ylabel('V')
plt.xlabel('ms')
plt.show()


#%%
import numpy as np
#import scipy as sp
from scipy import stats
from scipy import signal as si
import matplotlib.pyplot as plt

stim_time = 49
Fs = 4400
baseline = [145, 165]
sd_cutoff = 3
hard_cutoff = 150
peak_width = 4
peaks_in_signal = 4
diff_threshold = 0.009
gain = 1
x = np.linspace(0, total_time*1000, 300)  

fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks([])

for i in range(len(trials_post)):
    target_signal = trials_pre[i]
    sp, ep, peaks_final, latency, amplitude, Fmax, Fmin = peak_detection(stim_time=stim_time, target_signal = target_signal, sig_start=170, sig_end=295,
                             Fs=Fs, baseline=baseline, sd_cutoff=sd_cutoff, hard_cutoff=hard_cutoff,
                             peak_width=peak_width, peaks_in_signal=peaks_in_signal, diff_threshold=diff_threshold, gain=gain)
    
    plt.plot(x, target_signal, color = "gray", alpha = 0.5, linewidth=0.8)
    
    if len(peaks_final) != 0:
        plt.plot(x[sp], target_signal[sp],"o",color="blue")
        plt.plot(x[ep], target_signal[ep],"o",color="blue")
        plt.plot(x[sp:ep], target_signal[sp:ep], "green", linewidth = 2)
        plt.axvline(x[stim_time], color="k")
        
        #print("Amplitudes", amplitude)
    plt.title('Raw F-wave Traces EX001 pre')
    plt.ylabel('V')
    plt.xlabel('ms')
    
#%%
fig, ax = plt.subplots(figsize=(7,12))
ax.set_yticks([])

for i in range(len(trials_post)):
    target_signal = trials_post[i]
    sp, ep, peaks_final, latency, amplitude, Fmax, Fmin = peak_detection(stim_time=stim_time, target_signal = target_signal, sig_start=170, sig_end=295,
                             Fs=Fs, baseline=baseline, sd_cutoff=sd_cutoff, hard_cutoff=hard_cutoff,
                             peak_width=peak_width, peaks_in_signal=peaks_in_signal, diff_threshold=diff_threshold, gain=gain)
    
    plt.plot(x, target_signal, color = "gray", alpha = 0.5, linewidth=0.8)
    
    if len(peaks_final) != 0:
        plt.plot(x[sp], target_signal[sp],"o",color="blue")
        plt.plot(x[ep], target_signal[ep],"o",color="blue")
        plt.plot(x[sp:ep], target_signal[sp:ep], "green", linewidth = 2)
        plt.axvline(x[stim_time], color="k")
        
        #print("Amplitudes", amplitude)
    plt.title('Raw F-wave Traces EX001 post')
    plt.ylabel('V')
    plt.xlabel('ms')


#%%
def peak_detection(stim_time,target_signal,sig_start,sig_end,Fs,baseline,sd_cutoff,hard_cutoff,peak_width,peaks_in_signal,diff_threshold,gain):
        """Detects M and H reflex responses
        
        
        Keyword arguments:
        stim_time -- index at which stim occurs (default:400 @10khz sampling)
        target_signal -- input signal for detection (1-dimensional array)
        sig_start -- start index. The index by which the stimulation artifact is gone (default: 450 for M, 650 for H @10khz)
        sig_end -- end index. The index by which the reflex response is gone (default: 580 for M, 850 for H @10khz)
        Fs -- sampling frequency in hz (default: 10000 @10khz)
        baseline -- the start and end indices of a baseline signal, ie: no stimulation artifacts or reflex responses (default: [1,300] @10khz)
        sd_cutoff -- the threshold number of standard deviations away from the mean a signal peak must be (default: 200)
        hard_cutoff -- the min to max amplitude cutoff for a response to be considered (in uV) (default: 50) 
        peak_width -- the minimum width of the reflex response, in indices (default: 10 @10khz)
        peaks_in_signal -- number of peaks in the reflex response in the rectified signal (default: 2)
        diff_threshold -- the maximum slope use to find the start and endpoints of the peak, difference between two subsequent readings (default: .009, or .005)
        gain -- filter gain (default: 500)
        
        """
        diff_threshold = diff_threshold/gain
        peaks_to_fill = np.zeros(peaks_in_signal) # Zero array to fill with sorted peak indices
        signal = target_signal[sig_start:sig_end] # Truncates the signal by start and endpoints
        
        peaks, properties = si.find_peaks(abs(signal), height = stats.tstd(target_signal[baseline[0]:baseline[1]]) * sd_cutoff, width = peak_width) # Finds the peaks
        if peaks.size == 0: 
            #print("no peak detected")
            sp, ep, peaks_final, latency, amplitude, Fmax, Fmin = [],[],[],[],[],[],[]
            return sp, ep, peaks_final, latency, amplitude, Fmax, Fmin
        elif peaks.size == 1:
            #print("one peak only")
            sp, ep, peaks_final, latency, amplitude, Fmax, Fmin = [],[],[],[],[],[],[]
            return sp, ep, peaks_final, latency, amplitude, Fmax, Fmin
        elif peaks.size > 1: # Need at least two peaks
            #print("more than one peak")
            sorted_indices = properties["peak_heights"].argsort() # Gets returned peak indices based on height
            for i in range(peaks_in_signal):
                peaks_to_fill[i] = peaks[sorted_indices[(len(sorted_indices)-1)-i]] # Uses the sorted indices to get the final peaks in order
            try:
                sp = int(max(np.where(abs(np.diff((signal[0:int(min(peaks_to_fill))]), n=1)) < diff_threshold)[0]) + sig_start)
            except:
                #print("Start point could not be found. Try lowering difference threshold. Reverting to first signal peak.")
                sp = int(min(peaks_to_fill) + sig_start)
            try:
                ep = int(min(np.where(abs(np.diff((signal[int(max(peaks_to_fill)):len(signal)]), n=1)) < diff_threshold)[0] + int(max(peaks_to_fill))) + sig_start)
            except:
                #print("End point could not be found. Try lowering difference threshold. Reverting to last signal peak.")
                ep = int(max(peaks_to_fill) + sig_start)
            peaks_final = [min(peaks_to_fill) + sig_start, max(peaks_to_fill) + sig_start] # adds the signal start to account for the truncation
            latency = (peaks_final[0] - stim_time) * 1/Fs # Peak latency, in seconds
            #amplitude = round(target_signal[int(min(peaks_final))], 6) # Peak amplitude in volts, pre-gain
            amplitude = round(abs(max(target_signal[sp:ep])-min(target_signal[sp:ep])), 6)
            # final test to check if amplitude satisfies the cutoff
            # print(amplitude)
            Fmax = max(target_signal[sp:ep])
            Fmin = min(target_signal[sp:ep])
            if amplitude >= hard_cutoff * 1e-6:# Convert from microvolts to volts
                print(amplitude)
                return sp, ep, peaks_final, latency, amplitude, Fmax, Fmin
            else:
                sp, ep, peaks_final, latency, amplitude, Fmax, Fmin = [],[],[],[],[],[],[]
                return sp, ep, peaks_final, latency, amplitude, Fmax, Fmin

        # # Just visualization
        # plt.plot((target_signal))
        # plt.annotate("Latency: " + str(min(peaks_final) * 1/Fs) + " seconds\nAmplitude: " + str(round(target_signal[int(min(peaks_final))] / gain, 6)) + " volts", (min(peaks_final), target_signal[int(min(peaks_final))]))
        # plt.plot(sp, target_signal[sp],"o")
        # plt.plot(ep, target_signal[ep],"o")
        # plt.plot(range(sp,ep), target_signal[sp:ep], "r", linewidth = 2)


#%%












































#%%
#TESTING CODE

import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, argrelmin

# creation of data
# change point detection
# model = "l2"  # "l1", "rbf", "linear", "normal", "ar",...
# for i in range(len(trials)):
#     signal = trials[i]
#     signal_id = signal[150:]
#     x_id = x[150:]
    
#     algo = rpt.Binseg(model=model).fit(signal_id)
#     my_bkps = algo.predict(n_bkps=1)
    
#     # show results
#     #rpt.show.display(signal_id, my_bkps, figsize=(10, 6))
#     plt.show()
    
#     index = my_bkps[0]
#     start = index - 30
#     end = index + 20
    
#     plt.plot(x, signal, color = "gray", alpha = 0.5)
#     plt.plot(x_id[start:end], signal_id[start:end], color = "red")
peaks_list = []

x = np.linspace(0, total_time*1000, 150) 

for i in range(len(trials)):
    signal = trials[i]
    
    rest = signal_id[0:10]
    
    signal_id = signal[150:]
    x_id = x[150:]
    
    height = np.mean(signal_id) + (2*np.std(signal_id)) #height requirement for peak detection
    peaks, _ = find_peaks(signal_id, height=height)

    plt.plot(x, signal_id, "gray", alpha = 0.5) #plot trigger signal
    #plt.plot(peaks, signal_id[peaks], "x", markersize = 10, color = "red") #plot markers on peaks
    start = 0
    end = 0
    if len(peaks) != 0:
        start = peaks[0] - 30
        end = peaks[0] + 2
        valley = argrelmin(signal_id[start:end])
        if len(valley[0]) == 0:
            valley = [5]
 
        check = start + valley[0]
        #print(check)
        plt.plot(x[check], signal_id[check], 'o', markersize = 8, color = "purple")
        check = int(check)
        print(check)
        #print(check)
        
    plt.plot(x[start:end],(signal_id[start:end]),color = "blue")
    plt.plot(x[peaks], signal_id[peaks], "o", markersize = 8, color = "purple")
    #plt.plot(x[0:10],(signal_id[0:10]),color = "green")

from findpeaks import findpeaks
x = np.linspace(0, total_time*1000, 150) 

h = np.array(trials)
res = np.average(h, axis=0)

import peakdetect


for i in range(len(trials)):
    signal = trials[i] - res
    
    signal_id = signal[150:]
    fp = findpeaks(method='peakdetect', whitelist='peak', lookahead=30)
    
    results = fp.fit(signal_id)
    df = results['df']
    
    valley = df.index[df['valley'] == True].tolist()
    if len(valley) != 0:
        valley.pop(0)
        valley.pop()
    peak = df.index[df['peak'] == True].tolist()
    if len(peak) !=0:
        peak = peak.pop()
    print(peak)
    
    #value = signal_id[p]
    #fp.plot()
    plt.plot(x, signal_id, color = "gray", alpha = 0.5)
    plt.plot(x[peak], signal_id[peak], "o", color = "green", markersize = 8)
    plt.plot(x[valley], signal_id[valley], "v", color = "red", markersize = 8)
    
    
#27, 24, 19, 15, 13, 11, 8












